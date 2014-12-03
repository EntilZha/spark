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

import breeze.linalg.{DenseVector}
import org.apache.commons.math3.special.Gamma
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
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
  // Topic represents the currently assigned topic as the first field. The second field represents
  // the prior topic
  type Topic = Long

  class Posterior (docs:VertexRDD[Array[Int]], words: VertexRDD[Array[Int]])

  def currentTopic(topic:Topic): Int = {
    (topic >> 32).asInstanceOf[Int]
  }

  def oldTopic(topic:Topic): Int = {
    return topic.asInstanceOf[Int]
  }
  def combineTopics(currentTopic:Int, oldTopic:Int): Topic = {
    (currentTopic.asInstanceOf[Long] << 32) | (oldTopic & 0xffffffffL)
  }
  /**
   * Sums two factors together into a, then returns it. This increases memory efficiency
   * and reduces garbage collection.
   * @param a First factor
   * @param b Second factor
   * @return Sum of factors
   */
  def combineHistograms(a:Array[Int], b:Array[Int]): Array[Int] = {
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
   * @param topic topic to add
   * @return Result of adding topic into factor.
   */
  def combineTopicIntoHistogram(a:Array[Int], topic:Topic): Array[Int] = { a(LDA.currentTopic(topic)) += 1; a }

  /**
   * Creates a factor with topic added to it.
   * @param nTopics Number of topics
   * @param topic Topic to start with
   * @return New factor with topic added to it
   */
  def makeHistogram(nTopics: Int, topic:Topic): Array[Int] = {
    val f = new Array[Int](nTopics)
    f(LDA.currentTopic(topic)) += 1
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
   * @param totalHistBroadcast Total histogram of topics
   * @param nt Number of topics
   * @param alpha Parameter for dirichlet prior on per document topic distributions
   * @param beta Parameter for the dirichlet prior on per topic word distributions
   * @param nw Number of words in corpus
   * @return New topic for token/triplet
   */
  def sampleToken(gen:java.util.Random,
                  triplet:EdgeTriplet[Array[Int], Topic],
                  totalHistBroadcast:Broadcast[Array[Int]],
                  totalHistArgSorted:Broadcast[Array[Int]],
                  nt:Int,
                  alpha:Double,
                  beta:Double,
                  nw:Long): Topic = {
    val totalHist = totalHistBroadcast.value
    val wHist = triplet.srcAttr
    val dHist = triplet.dstAttr
    val oldTopic = LDA.currentTopic(triplet.attr)
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
    LDA.combineTopics(newTopic, oldTopic)
  }
  def fastSampleToken(gen:java.util.Random,
                      triplet:EdgeTriplet[Array[Int], Topic],
                      totalHistBroadcast:Broadcast[Array[Int]],
                      totalHistArgMinsBroadcast:Broadcast[Array[Int]],
                      nt:Int,
                      alpha:Double,
                      beta:Double,
                      nw:Long): Topic = {
    val topic = LDA.currentTopic(triplet.attr)
    val aCounts = triplet.dstAttr
    val bCounts = triplet.srcAttr
    val cCounts = totalHistBroadcast.value
    val cArgMins = totalHistArgMinsBroadcast.value
    aCounts(topic) -= 1
    bCounts(topic) -= 1
    cCounts(topic) -= 1
    var aSquareSum:Double = 0
    var bSquareSum:Double = 0
    for (i <- 0 until nt) {
      aSquareSum += math.pow(aCounts(i) + alpha, 2)
      bSquareSum += math.pow(bCounts(i) + alpha, 2)
    }
    val zBound = new Array[Double](nt)
    val sumP = new Array[Double](nt)
    var u = gen.nextDouble()
    for (k <- 0 until nt) {
      val a = aCounts(k) + alpha
      val b = bCounts(k) + beta
      val c = 1 / (cCounts(k) + beta * nw)
      val priorSumP = if (k == 0) 0 else sumP(k - 1)
      sumP(k) = priorSumP + a * b * c
      val aSubtractTerm = math.pow(a, 2)
      val bSubtractTerm = math.pow(b, 2)
      aSquareSum = if (aSquareSum - aSubtractTerm > 0) aSquareSum - aSubtractTerm else 0
      bSquareSum = if (bSquareSum - bSubtractTerm > 0) bSquareSum - bSubtractTerm else 0
      val cMin = cArgMins(k)
      val aNorm = math.sqrt(aSquareSum)
      val bNorm = math.sqrt(bSquareSum)
      val cNorm = 1 / (cMin + nw * beta)
      zBound(k) = sumP(k) + aNorm * bNorm * cNorm
      if (u * zBound(k) <= sumP(k)) {
        if (k == 0 || u * zBound(k) > priorSumP) {
          return LDA.combineTopics(k, topic)
        } else {
          u = (u * zBound(k - 1) - priorSumP) * zBound(k) / (zBound(k - 1) - zBound(k))
          for (t <- 0 until k) {
            if (sumP(t) >= u) {
              return LDA.combineTopics(t, topic)
            }
          }
        }
      }
    }
    throw new Exception("fast sample token failed")
  }

  /**
   * Given a list, first uses argsort. For each i from 0 to nt, the position is the position of the argmin
   * for elements in [i,nt)
   * @param c Global counts histogram
   * @return argmin list described above
   */
  def argminForShrinkingList(c:Array[Int], nt:Int): Array[Int] = {
    val cArgSort:Array[Int] = breeze.linalg.argsort(new DenseVector(c)).toArray
    var position = 0
    val argMins = new Array[Int](nt)
    argMins(0) = c(cArgSort(position))
    for (i <- 1 until nt) {
      var currArgMin = argMins(i - 1)
      if (currArgMin < i) {
        var continue = true
        while (continue && position < nt) {
          if (cArgSort(position) >= i) {
            currArgMin = cArgSort(position)
            continue = false
          }
          position += 1
        }
      }
      argMins(i) = c(currArgMin)
    }
    return argMins
  }
}

/**
 * LDA contains the model for topic modeling using Latent Dirichlet Allocation
 * @param tokens RDD of edges, transient to insure it doesn't get sent to workers
 * @param nTopics Number of topics
 * @param alpha Model parameter, governs sparsity in document-topic mixture
 * @param beta Model parameter, governs sparsity in word-topic mixture
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
  private var graph:Graph[Array[Int], Topic] = {
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
    val gTmp = Graph.fromEdgeTuples(renumbered, false).mapEdges({ (pid, iter) =>
        val gen = new java.util.Random(pid)
        iter.map(e => LDA.combineTopics(gen.nextInt(nT), 0))
    })
    // Compute the topic histograms (factors) for each word and document
    val newCounts = gTmp.mapReduceTriplets[Array[Int]](
      e => Iterator((e.srcId, makeHistogram(nT, e.attr)), (e.dstId, makeHistogram(nT, e.attr))),
      (a, b) => combineHistograms(a,b) )
    // Update the graph with the factors
    gTmp.outerJoinVertices(newCounts)({(_, _, newFactorOpt) => newFactorOpt.get }).cache
  }

  /**
   * Get the word vertices by filtering on non-negative vertices
   * @return Word vertices
   */
  def wordVertices: VertexRDD[Array[Int]] = graph.vertices.filter{ case (vid, c) => vid >= 0 }

  /**
   * Get the document vertices by filtering on negative vertices
   * @return Document vertices
   */
  def docVertices: VertexRDD[Array[Int]] = graph.vertices.filter{ case (vid, c) => vid < 0 }

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
    .aggregate(new Array[Int](nTopics))(LDA.combineTopicIntoHistogram(_, _), LDA.combineHistograms(_, _))

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
   * @param iterations Number of iterations to execute
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
      val totalHistArgSort:Array[Int] = breeze.linalg.argsort(new DenseVector(totalHist)).toArray
      val totalHistArgSortbcast = sc.broadcast(totalHistArgSort)

      // Shadowing because scala's closure capture would otherwise serialize the model object
      val a = alpha
      val b = beta
      val nt = nTopics
      val nw = nWords

      // Re-sample all the tokens
      var tempTimer:Long = System.nanoTime()
      val parts = graph.edges.partitions.size
      val interIter = internalIteration
      graph = graph.mapTriplets({(pid:PartitionID, iter:Iterator[EdgeTriplet[Array[Int], LDA.Topic]]) =>
        val gen = new java.util.Random(parts * interIter + pid)
        iter.map({ token =>
          LDA.sampleToken(gen, token, totalHistbcast, totalHistArgSortbcast, nt, a, b, nw)
        })
      }, TripletFields.All)
      if (loggingTime && i % loggingInterval == 0) {
        graph.cache.triplets.count()
        resampleTimes += System.nanoTime() - tempTimer
      }

      // Update the counts
      tempTimer = System.nanoTime()

      //val newCounts = graph.triplets
      //                   .flatMap(e => {Iterator((e.srcId, e.attr), (e.dstId, e.attr))})
      //                   .aggregateByKey(new Array[Int](nt))(LDA.combineTopicIntoHistogram(_, _), LDA.combineHistograms(_, _))
      // Stores a list of deltas per vertex. Each delta is a Topic, which packs two Ints into a Long
      // The left Int is the new topic, the right Int is the old topic
      val deltas = graph.aggregateMessages[ListBuffer[Topic]]({context =>
        val currentTopic = LDA.currentTopic(context.attr)
        val oldTopic = LDA.oldTopic(context.attr)
        if (currentTopic != oldTopic) {
          val message = new ListBuffer[LDA.Topic]()
          message += context.attr
          context.sendToDst(message)
          context.sendToSrc(message)
        }
      }, _ ++= _, TripletFields.EdgeOnly)

      graph = graph.outerJoinVertices(deltas)({(_, histogram, vertexDeltasOption) =>
        if (vertexDeltasOption.isDefined) {
          val vertexDeltas = vertexDeltasOption.get.iterator
          while (vertexDeltas.hasNext) {
            val topic = vertexDeltas.next()
            val currentTopic = LDA.currentTopic(topic)
            val oldTopic = LDA.oldTopic(topic)
            histogram(oldTopic) -= 1
            histogram(currentTopic) += 1
          }
        }
        histogram
      }).cache

      //graph = graph.outerJoinVertices(newCounts)({(_, _, newFactorOpt) => newFactorOpt.get }).cache
      if (loggingTime && i % loggingInterval == 0) {
        graph.cache.triplets.count()
        updateCountsTimes += System.nanoTime() - tempTimer
      }

      // Recompute the global counts (the actual action)
      tempTimer = System.nanoTime()
      totalHist = graph.edges.map(e => e.attr)
        .aggregate(new Array[Int](nt))(LDA.combineTopicIntoHistogram(_, _), LDA.combineHistograms(_, _))
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
  def topWords(k:Int): Array[Array[(Int, WordId)]] = {
    val nt = nTopics
    graph.vertices.filter({
      case (vid, c) => vid >= 0
    }).mapPartitions({ items =>
      val queues = Array.fill(nt)(new BoundedPriorityQueue[(Int, WordId)](k))
      for ((wordId, factor) <- items) {
        var t = 0
        while (t < nt) {
          val tpl: (Int, WordId) = (factor(t), wordId)
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
  def posterior: Posterior = {
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
      totalHist.map(v => Gamma.logGamma(v + nw * b)).sum +
      wordVertices.map({ case (id, f) => f.map(v => Gamma.logGamma(v + b)).sum})
                  .reduce(_ + _)
    val logPZ =
      nd * (Gamma.logGamma(nt * a) - nt * logAlpha) +
      docVertices.map({ case (id, f) =>
        f.map(v => Gamma.logGamma(v + a)).sum - Gamma.logGamma(nt * a + f.sum)
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
