/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.graphx.lib

import breeze.linalg.DenseVector
import org.apache.commons.math3.special.Gamma
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.graphx._
import org.apache.spark.graphx.PartitionStrategy.{CanonicalRandomVertexCut, EdgePartition1D, EdgePartition2D, RandomVertexCut}
import org.apache.spark.graphx.util.TimeTracker
import org.apache.spark.rdd.RDD
import org.apache.spark.util.BoundedPriorityQueue
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * LDA contains utility methods used in the LDA class. These are mostly
 * methods which will be serialized during computation so cannot be methods.
 */
object LDA {
  type DocId = VertexId
  type WordId = VertexId
  type TopicId = Int
  type Count = Int

  type Factor = Array[Count]

  class Posterior (docs:VertexRDD[Factor], words:VertexRDD[Factor])

  /**
   * Sums two factors together into a, then returns it. This increases memory efficiency
   * and reduces garbage collection.
   * @param a First factor
   * @param b Second factor
   * @return Sum of factors
   */
  def addEq(a:Factor, b:Factor):Factor = {
    assert(a.size == b.size)
    var i = 0
    while (i < a.size) {
      a(i) += b(i)
      i += 1
    }
    a
  }

  /**
   * Combines a topic into a factor
   * @param a Factor to add to
   * @param t topic to add
   * @return Result of adding topic into factor.
   */
  def addEq(a:Factor, t:TopicId):Factor = { a(t) += 1; a }

  /**
   * Creates a factor with topic added to it.
   * @param nTopics Number of topics
   * @param topic Topic to start with
   * @return New factor with topic added to it
   */
  def makeFactor(nTopics: Int, topic: TopicId): Factor = {
    val f = new Factor(nTopics)
    f(topic) += 1
    f
  }

  /**
   * Extracts the vocabulary from the RDD of tokens. Returns a Map from each word to its unique
   * number key, and an array indexable by that number key to the word
   * @param tokens RDD of tokens to create vocabulary from
   * @return array and map for looking up words from keys and keys from words.
   */
  def extractVocab(tokens:RDD[String]): (Array[String], mutable.Map[String, WordId]) = {
    val vocab = tokens.distinct().collect()
    var vocabLookup = mutable.Map[String, WordId]()
    for (i <- 0 to vocab.length - 1) {
      vocabLookup += (vocab(i) -> i)
    }
    (vocab, vocabLookup)
  }

  /**
   * Extracts edges from an RDD of documents. Each document is a single line/string
   * in the RDD
   * @param lines RDD of documents
   * @param vocab Vocabulary in the documents
   * @param vocabLookup Vocabulary lookup in the documents
   * @param delimiter Delimiter to split on
   * @return RDD of edges between words and documents representing tokens.
   */
  def edgesFromTextDocLines(lines:RDD[String],
                            vocab:Array[String],
                            vocabLookup:mutable.Map[String, WordId],
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

  /**
   * Re-samples the given token/triplet to a new topic
   * @param gen Random number generator
   * @param triplet Token to re-sample
   * @param totalHist Total histogram of topics
   * @param nt Number of topics
   * @param alpha Parameter for dirichlet prior on per document topic distributions
   * @param beta Parameter for the dirichlet prior on per topic word distributions
   * @param nw Number of words in corpus
   * @return New topic for token/triplet
   */
  def sampleToken(gen:java.util.Random,
                  triplet:EdgeTriplet[Factor, TopicId],
                  totalHist:LDA.Factor,
                  nt:Int,
                  alpha:Double,
                  beta:Double,
                  nw:Long):TopicId = {

    val wHist:Factor = triplet.srcAttr
    val dHist:Factor = triplet.dstAttr
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
    var newTopic = 0
    var cumsum = conditional(newTopic)
    while (cumsum < u) {
      newTopic += 1
      cumsum += conditional(newTopic)
    }
    newTopic
  }
  def fastSampleToken(gen:java.util.Random,
                      triplet:EdgeTriplet[Factor, TopicId],
                      totalHist:LDA.Factor,
                      nt:Int,
                      alpha:Double,
                      beta:Double,
                      nw:Long):TopicId = {
    val preA = triplet.dstAttr
    val preB = triplet.srcAttr
    val preC = totalHist
    val topic = triplet.attr
    val sumP = DenseVector.zeros[Double](nt)
    val u = gen.nextDouble()
    for (k <- 1 until nt) {
      val offset = if (k == topic) 1 else 0
      val a = preA(k) - offset + alpha
      val b = preB(k) - offset + beta
      val c = 1 / (preC(k) - offset + beta * nw)
      sumP(k) = sumP(k - 1) + a * b * c
    }
    return 0
  }
}

/**
 * LDA contains the model for topic modeling using Latent Dirichlet Allocation
 * @param tokens RDD of edges, transient to insure it doesn't get sent to workers
 * @param nTopics Number of topics
 * @param alpha Model parameter
 * @param beta Model parameter
 * @param loggingInterval Interval for logging
 * @param loggingLikelihood If true, log the likelihood
 * @param loggingTime if true, log the runtime of each component
 */
class LDA(@transient val tokens: RDD[(LDA.WordId, LDA.DocId)],
          val nTopics: Int = 100,
          val alpha: Double = 0.1,
          val beta: Double = 0.1,
          val loggingInterval: Int = 0,
          val loggingLikelihood: Boolean = false,
          val loggingTime: Boolean = false) extends Serializable with Logging {
  import org.apache.spark.graphx.lib.LDA._

  val timer = new TimeTracker()
  timer.start("setup")
  logInfo("Starting LDA setup")
  private val sc = tokens.sparkContext

  /**
   * The bipartite terms by document graph.
   */
  private var graph:Graph[Factor, TopicId] = {
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
  }

  /**
   * Get the word vertices by filtering on non-negative vertices
   * @return Word vertices
   */
  def wordVertices: VertexRDD[LDA.Factor] = graph.vertices.filter{ case (vid, c) => vid >= 0 }

  /**
   * Get the document vertices by filtering on negative vertices
   * @return Document vertices
   */
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

  /**
   * List to track time spent doing Gibbs sampling
   */
  val resampleTimes = new ListBuffer[Long]()

  /**
   * List to track time spent updating the counts on the graph
   */
  val updateCountsTimes = new ListBuffer[Long]()

  /**
   * List to track time spent updating the global topic histogram
   */
  val globalCountsTimes = new ListBuffer[Long]()

  /**
   * List to track negative log likelihood
   */
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
   * Trains the model by iterating nIter times
   * @param nIter Number of iterations to execute
   */
  def train(iterations:Int) {
    // Run the sampling
    timer.start("run")
    logInfo("Starting LDA Iterations...")
    for (i <- 0 until iterations) {
      logInfo(s"Iteration $i of $iterations...")

      // Log the negative log likelihood
      if (loggingLikelihood && i % loggingInterval == 0) {
        val likelihood = logLikelihood()
        likelihoods += likelihood
      }
      // Broadcast the topic histogram
      val totalHistbcast = sc.broadcast(totalHist)

      // Shadowing because scala's closure capture would otherwise serialize the model object
      val a = alpha
      val b = beta
      val nt = nTopics
      val nw = nWords

      // Re-sample all the tokens
      var tempTimer:Long = System.nanoTime()
      val parts = graph.edges.partitions.size
      val interIter = internalIteration
      graph = graph.mapTriplets({(pid:PartitionID, iter:Iterator[EdgeTriplet[LDA.Factor, LDA.TopicId]]) =>
        val gen = new java.util.Random(parts * interIter + pid)
        iter.map({ token =>
          LDA.sampleToken(gen, token, totalHistbcast.value, nt, a, b, nw)
        })
      }, TripletFields.All)
      if (loggingTime && i % loggingInterval == 0) {
        graph.cache.triplets.count()
        resampleTimes += System.nanoTime() - tempTimer
      }

      // Update the counts
      tempTimer = System.nanoTime()

      val newCounts = graph.triplets
                         .flatMap(e => {Iterator((e.srcId, e.attr), (e.dstId, e.attr))})
                         .aggregateByKey(new Factor(nt))(LDA.addEq(_, _), LDA.addEq(_, _))
      //val newCounts = graph.mapReduceTriplets[Factor](
      //  e => Iterator((e.srcId, makeFactor(nt, e.attr)), (e.dstId, makeFactor(nt, e.attr))),
      //  (a, b) => { addEq(a,b); a } )
      graph = graph.outerJoinVertices(newCounts) { (_, _, newFactorOpt) => newFactorOpt.get }.cache
      if (loggingTime && i % loggingInterval == 0) {
        graph.cache.triplets.count()
        updateCountsTimes += System.nanoTime() - tempTimer
      }

      // Recompute the global counts (the actual action)
      tempTimer = System.nanoTime()
      totalHist = graph.edges.map(e => e.attr)
        .aggregate(new Factor(nt))(LDA.addEq(_, _), LDA.addEq(_, _))
      assert(totalHist.sum == nTokens)
      if (loggingTime && i % loggingInterval == 0) {
        globalCountsTimes += System.nanoTime() - tempTimer
      }

      internalIteration += 1
    }
    // Log the final results of training
    val likelihood = logLikelihood()
    likelihoods += likelihood
    timer.stop("run")
    logInfo("LDA Finishing...")
    logPerformanceStatistics()
  }

  /**
   * Creates an object holding the top counts. The first array is of size number
   * of topics. It contains a list of k elements representing the top words for that topic
   * @param k Number of top words to output
   * @return object with top counts for each word.
   */
  def topWords(k:Int): Array[Array[(Count, WordId)]] = {
    val nt = nTopics
    graph.vertices.filter({
      case (vid, c) => vid >= 0
    }).mapPartitions({ items =>
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
    }).reduce({ (q1, q2) =>
      q1.zip(q2).foreach({ case (a,b) => a ++= b })
      q1
    }).map(q => q.toArray)
  }

  /**
   * Creates the posterior distribution for sampling from the vertices
   * @return Posterior distribution
   */
  def posterior:Posterior = {
    graph.cache()
    val words = graph.vertices.filter({ case (vid, _) => vid >= 0 })
    val docs =  graph.vertices.filter({ case (vid,_) => vid < 0 })
    new LDA.Posterior(words, docs)
  }

  /**
   * Calculates the log likelihood according to:
   * \mathcal{L}( w | z) & = T * \left( \log\Gamma(W * \beta) - W * \log\Gamma(\beta) \right) + \\
    & \sum_{t} \left( \left(\sum_{w} \log\Gamma(N_{wt} + \beta)\right) -
           \log\Gamma\left( W * \beta + \sum_{w} N_{wt}  \right) \right) \\
    & = T * \left( \log\Gamma(W * \beta) - W * \log\Gamma(\beta) \right) -
        \sum_{t} \log\Gamma\left( W * \beta + N_{t}  \right) + \\
    & \sum_{w} \sum_{t} \log\Gamma(N_{wt} + \beta)   \\
    \\
    \mathcal{L}(z) & = D * \left(\log\Gamma(T * \alpha) - T * \log\Gamma(\alpha) \right) + \\
    & \sum_{d} \left( \left(\sum_{t}\log\Gamma(N_{td} + \alpha)\right) -
        \log\Gamma\left( T * \alpha + \sum_{t} N_{td} \right) \right) \\
    \\
    \mathcal{L}(w,z) & = \mathcal{L}(w | z) + \mathcal{L}(z)\\

    N_{td} =\text{number of tokens with topic t in document d}\\
    N_{wt} =\text{number of tokens with topic t for word w}
   */
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
      wordVertices.map({ case (id, f) => f.map(v => Gamma.logGamma(v + b)).reduce(_ + _)})
                  .reduce(_ + _)
    val logPZ =
      nd * (Gamma.logGamma(nt * a) - nt * logAlpha) +
      docVertices.map({ case (id, f) =>
        f.map(v => Gamma.logGamma(v + a)).reduce(_ + _) - Gamma.logGamma(nt * a + f.reduce(_ + _))
      }).reduce(_ + _)
    logPWGivenZ + logPZ
  }

  /**
   * Logs the final machine performance and ML performance to INFO
   */
  def logPerformanceStatistics() = {
    val resampleTime = if (loggingTime) resampleTimes.reduce(_ + _) / 1e9 else 0
    val updateCountsTime = if (loggingTime) updateCountsTimes.reduce(_ + _) / 1e9 else 0
    val globalCountsTime = if (loggingTime) globalCountsTimes.reduce(_ + _) / 1e9 else 0
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
    if (loggingTime) {
      logInfo(s"Resample Time: $resampleTime")
      logInfo(s"Update Counts Time: $updateCountsTime")
      logInfo(s"Global Counts Time: $globalCountsTime")
    }
    logInfo("Machine Learning Performance")
    if (loggingLikelihood) {
      logInfo(s"Likelihoods: $likelihoodString")
    }
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

}
