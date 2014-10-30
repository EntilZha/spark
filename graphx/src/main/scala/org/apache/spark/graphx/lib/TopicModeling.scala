package org.apache.spark.graphx.lib

import org.apache.commons.math3.special.Gamma
import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.broadcast._
import org.apache.spark.graphx._
import org.apache.spark.graphx.PartitionStrategy.{CanonicalRandomVertexCut, EdgePartition1D, EdgePartition2D, RandomVertexCut}
import org.apache.spark.graphx.util.TimeTracker
import org.apache.spark.rdd.RDD
import org.apache.spark.util.BoundedPriorityQueue
import org.apache.log4j.{LogManager, Level, Logger}

import scala.collection.mutable.ListBuffer

object LDA {
  type DocId = VertexId
  type WordId = VertexId
  type TopicId = Int
  type Count = Int

  type Factor = Array[Count]

  class Posterior (docs: VertexRDD[Factor], words: VertexRDD[Factor])

  // For memory efficiency/garbage collection, don't allocate a new Factor, return one of the originals
  def addEq(a: Factor, b: Factor): Factor = {
    assert(a.size == b.size)
    var i = 0
    while (i < a.size) {
      a(i) += b(i)
      i += 1
    }
    a
  }

  // For memory efficiency/garbage collection, don't allocate a new Factor, return the original
  def addEq(a: Factor, t: TopicId): Factor = { a(t) += 1; a }

  def makeFactor(nTopics: Int, topic: TopicId): Factor = {
    val f = new Factor(nTopics)
    f(topic) += 1
    f
  }
  def extractVocab(tokens:RDD[String]): (Array[String], scala.collection.mutable.Map[String, WordId]) = {
    val vocab = tokens.distinct().collect()
    var vocabLookup = scala.collection.mutable.Map[String, WordId]()
    for (i <- 0 to vocab.length - 1) {
      vocabLookup += (vocab(i) -> i)
    }
    (vocab, vocabLookup)
  }
  def edgesFromTextDocLines(lines:RDD[String],
                            vocab:Array[String],
                            vocabLookup:Map[String, WordId],
                            delimiter:String=" "): RDD[(LDA.WordId, LDA.DocId)] = {
    val sc = lines.sparkContext
    val numDocs = lines.count()
    val docIds:RDD[DocId] = sc.parallelize((0L until numDocs).toArray)
    val docsWithIds = lines.zip(docIds)
    val edges:RDD[(WordId, DocId)] = docsWithIds.flatMap{ case (line:String, docId:DocId) =>
      val words = line.split(delimiter)
      val docEdges = words.map(word => (vocabLookup(word), docId))
      docEdges
    }
    edges
  }
  def sampleToken(gen:java.util.Random,
                  triplet:EdgeTriplet[Factor, TopicId],
                  totalHistBroadCast:Broadcast[LDA.Factor],
                  nt:Int,
                  alpha:Double,
                  beta:Double,
                  nw:Long): Int = {

    val wHist:Array[Count] = triplet.srcAttr
    val dHist:Array[Count] = triplet.dstAttr
    val totalHist = totalHistBroadCast.value
    val oldTopic = triplet.attr
    assert(wHist(oldTopic) > 0)
    assert(dHist(oldTopic) > 0)
    assert(totalHist(oldTopic) > 0)
    // Construct the conditional
    val conditional = new Array[Double](nt)
    var t = 0
    var conditionalSum = 0.0
    while (t < conditional.size) {
      val cavityOffset = if (t == oldTopic) 1 else 0
      val w = wHist(t) - cavityOffset
      val d = dHist(t) - cavityOffset
      val total = totalHist(t) - cavityOffset
      conditional(t) = (alpha + d) * (beta + w) / (beta * nw + total)
      conditionalSum += conditional(t)
      t += 1
    }
    assert(conditionalSum > 0.0)
    // Generate a random number between [0, conditionalSum)
    val u = gen.nextDouble() * conditionalSum
    assert(u < conditionalSum)
    // Draw the new topic from the multinomial
    t = 0
    var cumsum = conditional(t)
    while (cumsum < u) {
      t += 1
      cumsum += conditional(t)
    }
    val newTopic = t
    return newTopic
  }
}

/**
 * LDA contains the model for topic modeling using Latent Dirichlet Allocation
 * @param tokens RDD of edges
 * @param nTopics Number of topics
 * @param alpha Model parameter
 * @param beta Model parameter
 * @param logIter Interval for logging, if set to 0 only logs at model return
 */
class LDA(@transient val tokens: RDD[(LDA.WordId, LDA.DocId)],
          val nTopics: Int = 100,
          val alpha: Double = 0.1,
          val beta: Double = 0.1,
          val logIter: Int = 0) extends Serializable with Logging {
  import org.apache.spark.graphx.lib.LDA._

  val timer = new TimeTracker()
  timer.start("setup")
  logInfo("Starting LDA setup")
  private val sc = tokens.sparkContext

  /**
   * The bipartite terms by document graph.
   */
  private var graph: Graph[Factor, TopicId] = {
    // To setup a bipartite graph it is necessary to ensure that the document and
    // word ids are in a different namespace
    val renumbered = tokens.map { case (wordId, docId) =>
      assert(wordId >= 0)
      assert(docId >= 0)
      val newDocId: DocId = -(docId + 1L)
      (wordId, newDocId)
    }
    val nT = nTopics
    // Sample the tokens
    val gTmp = Graph.fromEdgeTuples(renumbered, false).mapEdges { (pid, iter) =>
        val gen = new java.util.Random(pid)
        iter.map(e => gen.nextInt(nT))
      }
    // Compute the topic histograms (factors) for each word and document
    val newCounts = gTmp.mapReduceTriplets[Factor](
      e => Iterator((e.srcId, makeFactor(nT, e.attr)), (e.dstId, makeFactor(nT, e.attr))),
      (a, b) => addEq(a,b) )
    // Update the graph with the factors
    gTmp.outerJoinVertices(newCounts) { (_, _, newFactorOpt) => newFactorOpt.get }.cache
    // Trigger computation of the topic counts
  }

  def wordVertices: VertexRDD[LDA.Factor] = graph.vertices.filter{ case (vid, c) => vid >= 0 }
  def docVertices: VertexRDD[LDA.Factor] = graph.vertices.filter{ case (vid, c) => vid < 0 }
  /**
   * The number of unique words in the corpus
   */
  val nWords = wordVertices.count()

  /**
   * The number of documents in the corpus
   */
  val nDocs = docVertices.count()

  /**
   * The number of tokens
   */
  val nTokens = graph.edges.count()

  /**
   * The total counts for each topic
   */
  var totalHist = graph.edges.map(e => e.attr)
    .aggregate(new Factor(nTopics))(LDA.addEq(_, _), LDA.addEq(_, _))
  assert(totalHist.sum == nTokens)

  val broadcastTimes = new ListBuffer[Long]()
  val resampleTimes = new ListBuffer[Long]()
  val updateCountsTimes = new ListBuffer[Long]()
  val globalCountsTimes = new ListBuffer[Long]()
  val likelihoods = new ListBuffer[Double]()

  /**
   * The internal iteration tracks the number of times the random number
   * generator was created.  In constructing the graph the generated is created
   * once and then once for each iteration
   */
  private var internalIteration = 1
  logInfo("LDA setup finished")
  timer.stop("setup")

  /**
   * Run the gibbs sampler
   * @param nIter
   * @return
   */
  def iterate(nIter: Int = 1) {
    // Run the sampling
    timer.start("run")
    logInfo("Starting LDA Iterations...")
    for (i <- 0 until nIter) {
      logInfo(s"Iteration $i of $nIter...")
      if (logIter != 0 && i % logIter == 0) {
        val likelihood = logLikelihood()
        likelihoods += likelihood
      }
      var tempTimer:Long = 0
      // Broadcast the topic histogram
      tempTimer = System.nanoTime()
      val totalHistbcast = sc.broadcast(totalHist)
      if (logIter != 0) {
        broadcastTimes += System.nanoTime() - tempTimer
      }
      // Shadowing because scala's closure capture is an abomination
      val a = alpha
      val b = beta
      val nt = nTopics
      val nw = nWords

      // Resample all the tokens
      tempTimer = System.nanoTime()
      val parts = graph.edges.partitions.size
      val interIter = internalIteration
      graph = graph.mapTriplets { (pid, iter) =>
        val gen = new java.util.Random(parts * interIter + pid)
        iter.map { token =>
          LDA.sampleToken(gen, token, totalHistbcast, nt, a, b, nw)
        }
      }
      if (logIter != 0) {
        resampleTimes += System.nanoTime() - tempTimer
      }

      // Update the counts
      tempTimer = System.nanoTime()
      val newCounts = graph.mapReduceTriplets[Factor](
        e => Iterator((e.srcId, makeFactor(nt, e.attr)), (e.dstId, makeFactor(nt, e.attr))),
        (a, b) => { addEq(a,b); a } )
      graph = graph.outerJoinVertices(newCounts) { (_, _, newFactorOpt) => newFactorOpt.get }.cache
      if (logIter != 0) {
        updateCountsTimes += System.nanoTime() - tempTimer
      }

      // Recompute the global counts (the actual action)
      tempTimer = System.nanoTime()
      totalHist = graph.edges.map(e => e.attr)
        .aggregate(new Factor(nt))(LDA.addEq(_, _), LDA.addEq(_, _))
      assert(totalHist.sum == nTokens)
      if (logIter != 0) {
        globalCountsTimes += System.nanoTime() - tempTimer
      }

      internalIteration += 1
    }
    if (logIter != 0) {
      val likelihood = logLikelihood()
      likelihoods += likelihood
    }
    timer.stop("run")
    logInfo("LDA Finishing...")
    logPerformanceStatistics()
  }

  def topWords(k: Int): Array[Array[(Count, WordId)]] = {
    val nt = nTopics
    graph.vertices.filter {
      case (vid, c) => vid >= 0
    }.mapPartitions { items =>
      val queues = Array.fill(nt)(new BoundedPriorityQueue[(Count, WordId)](k))
      for ((wordId, factor) <- items) {
        var t = 0
        while (t < nt) {
          val tpl: (Count, WordId) = (factor(t), wordId)
          queues(t) += tpl
          t += 1
        }
      }
      Iterator(queues)
    }.reduce { (q1, q2) =>
      q1.zip(q2).foreach { case (a,b) => a ++= b }
      q1
    }.map ( q => q.toArray )
  } // end of TopWords

  def posterior: Posterior = {
    graph.cache()
    val words = graph.vertices.filter { case (vid, _) => vid >= 0 }
    val docs =  graph.vertices.filter { case (vid,_) => vid < 0 }
    new LDA.Posterior(words, docs)
  }
  def logLikelihood(): Double = {
    val nw = nWords
    val nt = nTopics
    val nd = nDocs
    val a = alpha
    val b = beta
    val logAlpha = Gamma.logGamma(a)
    val logBeta = Gamma.logGamma(b)
    val logPWGivenZ =
      nTopics * (Gamma.logGamma(nw * b) - nw * logBeta) -
      totalHist.map(v => Gamma.logGamma(v + nw * b)).reduce(_ + _) +
      wordVertices.map({ case (id, f) => f.map(v => Gamma.logGamma(v + b)).reduce(_ + _)}).reduce(_ + _)
    val logPZ =
      nd * (Gamma.logGamma(nt * a) - nt * logAlpha) +
      docVertices.map({ case (id, f) =>
        f.map(v => Gamma.logGamma(v + a)).reduce(_ + _) - Gamma.logGamma(nt * a + f.reduce(_ + _))
      }).reduce(_ + _)
    logPWGivenZ + logPZ
  }

  def logPerformanceStatistics() = {
    val broadcastTime = broadcastTimes.reduce(_ + _) / 1e9
    val resampleTime = resampleTimes.reduce(_ + _) / 1e9
    val updateCountsTime = updateCountsTimes.reduce(_ + _) / 1e9
    val globalCountsTime = globalCountsTimes.reduce(_ + _) / 1e9
    val likelihoodList = likelihoods.toList
    val likelihoodString = likelihoodList.mkString(",")
    val finalLikelihood = likelihoodList.last
    logInfo("LDA Model Parameters and Information")
    logInfo(s"Number of Documents: $nDocs")
    logInfo(s"Number of Words: $nWords")
    logInfo(s"Number of Tokens: $nTokens")
    logInfo("Running Time Statistics")
    logInfo(s"Setup: $setupTime s")
    logInfo(s"Run: $runTime s")
    logInfo(s"Total: $totalTime s")
    logInfo(s"Broadcast Time: $broadcastTime")
    logInfo(s"Resample Time: $resampleTime")
    logInfo(s"Update Counts Time: $updateCountsTime")
    logInfo(s"Global Counts Time: $globalCountsTime")
    logInfo("Machine Learning Performance")
    logInfo(s"Likelihoods: $likelihoodString")
    logInfo(s"Final Log Likelihood: $finalLikelihood")
  }

  private def setupTime: Double = {
    timer.getSeconds("setup")
  }
  private def runTime: Double = {
    timer.getSeconds("run")
  }
  private def totalTime: Double = {
    timer.getSeconds("setup") + timer.getSeconds("run")
  }

} // end of TopicModeling



object TopicModeling {
  def main(args: Array[String]) {
    val host = args(0)
    val options =  args.drop(1).map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }

    var tokensFile = ""
    var dictionaryFile = ""
    var numVPart = 4
    var numEPart = 4
    var partitionStrategy: Option[PartitionStrategy] = None
    var nIter = 50
    var nTopics = 10
    var alpha = 0.1
    var beta  = 0.1

    def pickPartitioner(v: String): PartitionStrategy = v match {
      case "RandomVertexCut" => RandomVertexCut
      case "EdgePartition1D" => EdgePartition1D
      case "EdgePartition2D" => EdgePartition2D
      case "CanonicalRandomVertexCut" => CanonicalRandomVertexCut
      case _ => throw new IllegalArgumentException("Invalid Partition Strategy: " + v)
    }

    options.foreach{
      case ("tokens", v) => tokensFile = v
      case ("dictionary", v) => dictionaryFile = v
      case ("numVPart", v) => numVPart = v.toInt
      case ("numEPart", v) => numEPart = v.toInt
      case ("partStrategy", v) => partitionStrategy = Some(pickPartitioner(v))
      case ("niter", v) => nIter = v.toInt
      case ("ntopics", v) => nTopics = v.toInt
      case ("alpha", v) => alpha = v.toDouble
      case ("beta", v) => beta = v.toDouble
      case (opt, _) => throw new IllegalArgumentException("Invalid option: " + opt)
    }

    println("Tokens:     " + tokensFile)
    println("Dictionary: " + dictionaryFile)

    // def setLogLevels(level: org.apache.log4j.Level, loggers: TraversableOnce[String]) = {
    //   loggers.map{
    //     loggerName =>
    //       val logger = org.apache.log4j.Logger.getLogger(loggerName)
    //     val prevLevel = logger.getLevel()
    //     logger.setLevel(level)
    //     loggerName -> prevLevel
    //   }.toMap
    // }
    // setLogLevels(org.apache.log4j.Level.DEBUG, Seq("org.apache.spark"))


    val serializer = "org.apache.spark.serializer.KryoSerializer"
    System.setProperty("spark.serializer", serializer)
    //System.setProperty("spark.shuffle.compress", "false")
    System.setProperty("spark.kryo.registrator", "org.apache.spark.graph.GraphKryoRegistrator")
    val sc = new SparkContext(host, "LDA(" + tokensFile + ")")

    val rawTokens: RDD[(LDA.WordId, LDA.DocId)] =
      sc.textFile(tokensFile, numEPart).flatMap { line =>
      val lineArray = line.split("\\s+")
      if(lineArray.length != 3) {
        println("Invalid line: " + line)
        assert(false)
      }
      val termId = lineArray(0).trim.toLong
      val docId = lineArray(1).trim.toLong
      assert(termId >= 0)
      assert(docId >= 0)
      val count = lineArray(2).trim.toInt
      assert(count > 0)
      //Iterator((termId, docId))
      Iterator.fill(count)((termId, docId))
    }

    val dictionary =
      if (!dictionaryFile.isEmpty) {
        scala.io.Source.fromFile(dictionaryFile).getLines.toArray
      } else {
        Array.empty
      }

    val model = new LDA(rawTokens, nTopics, alpha, beta)

    for(iter <- 0 until nIter) {
      model.iterate(1)
      val topWords = model.topWords(5)
      for (queue <- topWords) {
        println("word list: ")
        if (!dictionary.isEmpty) {
          queue.foreach(t => println("\t(" + t._1 + ", " + dictionary(t._2.toInt - 1) + ")"))
        } else {
          queue.foreach(t => println("\t" + t.toString))
        }
      }
      println("Sampled iteration: " + iter.toString)
    }

    sc.stop()

  }
} // end of TopicModeling object

