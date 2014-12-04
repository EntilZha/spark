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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.Logging
import org.apache.spark.graphx._
import org.apache.spark.graphx.util.TimeTracker
import org.apache.spark.rdd.RDD
import org.apache.spark.util.BoundedPriorityQueue
import scala.collection.mutable

/**
 * LDA contains utility methods used in the LDA class. These are mostly
 * methods which will be serialized during computation so cannot be methods.
 */
object LDA {
  type DocId = VertexId
  type WordId = VertexId
  /**
   * Topic Long is composed of two Ints representint topics.
   * The left Int represents the current topic, right Int represents prior topic.
   */
  type Topic = Long

  case class Histogram(counts:Counts, argsort:Argsort) extends Serializable
  type Counts = Array[Int]
  type Argsort = Array[Int]

  class Posterior(docs: VertexRDD[Histogram], words: VertexRDD[Histogram])

  def currentTopic(topic: Topic): Int = {
    (topic >> 32).asInstanceOf[Int]
  }

  def oldTopic(topic: Topic): Int = {
    topic.asInstanceOf[Int]
  }
  def combineTopics(currentTopic: Int, oldTopic: Int): Topic = {
    (currentTopic.asInstanceOf[Long] << 32) | (oldTopic & 0xffffffffL)
  }
  /**
   * Sums two factors together into a, then returns it. This increases memory efficiency
   * and reduces garbage collection.
   * @param a First factor
   * @param b Second factor
   * @return Sum of factors
   */
  def combineCounts(a: Counts, b: Counts): Counts = {
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
  def combineTopicIntoCounts(a: Counts, topic: Topic): Counts = {
    a(LDA.currentTopic(topic)) += 1
    a
  }

  def combineDeltaIntoCounts(a: Counts, topic: Topic): Counts = {
    a(LDA.oldTopic(topic)) -= 1
    a(LDA.currentTopic(topic)) += 1
    a
  }

  /**
   * Creates a factor with topic added to it.
   * @param nTopics Number of topics
   * @param topic Topic to start with
   * @return New factor with topic added to it
   */
  def makeCountsFromTopic(nTopics: Int, topic: Topic): Counts = {
    val counts = new Counts(nTopics)
    counts(LDA.currentTopic(topic)) += 1
    counts
  }

  def makeHistogramFromCounts(counts:Counts): Histogram = {
    val argsort:Argsort = breeze.linalg.argsort(new DenseVector(counts)).toArray
    Histogram(counts, argsort)
  }

  /**
   * Extracts the vocabulary from the RDD of tokens. Returns a Map from each word to its unique
   * number key, and an array indexable by that number key to the word
   * @param tokens RDD of tokens to create vocabulary from
   * @return array and map for looking up words from keys and keys from words.
   */
  def extractVocab(tokens: RDD[String]): (Array[String], mutable.Map[String, WordId]) = {
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
  def edgesFromTextDocLines(lines: RDD[String],
                            vocab: Array[String],
                            vocabLookup: mutable.Map[String, WordId],
                            delimiter: String=" "): RDD[(LDA.WordId, LDA.DocId)] = {
    val sc = lines.sparkContext
    val numDocs = lines.count()
    val docIds: RDD[DocId] = sc.parallelize((0L until numDocs).toArray)
    val docsWithIds = lines.zip(docIds)
    val edges: RDD[(WordId, DocId)] = docsWithIds.flatMap{ case (line: String, docId: DocId) =>
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
   * @param totalHistogramBroadcast Total histogram of topics
   * @param nt Number of topics
   * @param alpha Parameter for dirichlet prior on per document topic distributions
   * @param beta Parameter for the dirichlet prior on per topic word distributions
   * @param nw Number of words in corpus
   * @return New topic for token/triplet
   */
  def sampleToken(gen: java.util.Random,
                  triplet: EdgeTriplet[Histogram, Topic],
                  totalHistogramBroadcast: Broadcast[Histogram],
                  nt: Int,
                  alpha: Double,
                  beta: Double,
                  nw: Long): Topic = {
    val totalCounts = totalHistogramBroadcast.value.counts
    val wHist = triplet.srcAttr.counts
    val dHist = triplet.dstAttr.counts
    val oldTopic = LDA.currentTopic(triplet.attr)
    assert(wHist(oldTopic) > 0)
    assert(dHist(oldTopic) > 0)
    assert(totalCounts(oldTopic) > 0)
    // Construct the conditional
    val conditional = new Array[Double](nt)
    var t = 0
    var conditionalSum = 0.0
    while (t < conditional.size) {
      val cavityOffset = if (t == oldTopic) 1 else 0
      val w = wHist(t) - cavityOffset
      val d = dHist(t) - cavityOffset
      val total = totalCounts(t) - cavityOffset
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

  def countWithoutTopic(counts:Counts, index: Int, topic:Int): Int = {
    if (index == topic) {
      counts(index) - 1
    } else {
      counts(index)
    }
  }

  def fastSampleToken(gen: java.util.Random,
                      triplet: EdgeTriplet[Histogram, Topic],
                      totalHistogramBroadcast: Broadcast[Histogram],
                      nt: Int,
                      alpha: Double,
                      beta: Double,
                      nw: Long): Topic = {
    val topic = LDA.currentTopic(triplet.attr)
    val totalHistogram = totalHistogramBroadcast.value
    val aCounts = triplet.dstAttr.counts
    val bCounts = triplet.srcAttr.counts
    val cCounts = totalHistogram.counts
    val cArgMins = totalHistogram.argsort
    var aSquareSum: Double = 0
    var bSquareSum: Double = 0
    for (i <- 0 until nt) {
      aSquareSum += math.pow(countWithoutTopic(aCounts, i, topic) + alpha, 2)
      bSquareSum += math.pow(countWithoutTopic(bCounts, i, topic) + alpha, 2)
    }
    val zBound = new Array[Double](nt)
    val sumP = new Array[Double](nt)
    var u = gen.nextDouble()
    for (k <- 0 until nt) {
      val a = countWithoutTopic(aCounts, k, topic) + alpha
      val b = countWithoutTopic(bCounts, k, topic) + beta
      val c = 1 / (countWithoutTopic(cCounts, k, topic) + beta * nw)
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
  def argminForShrinkingList(c: Array[Int], nt: Int): Array[Int] = {
    val cArgSort: Array[Int] = breeze.linalg.argsort(new DenseVector(c)).toArray
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
    argMins
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
  private var graph: Graph[Histogram, Topic] = {
    // To setup a bipartite graph it is necessary to ensure that the document and
    // word ids are in a different namespace
    val renumbered = tokens.map({ case (wordId, docId) =>
      assert(wordId >= 0)
      assert(docId >= 0)
      val newDocId: DocId = -(docId + 1L)
      (wordId, newDocId)
    })
    val nT = nTopics
    // Sample the tokens
    val gTmp = Graph.fromEdgeTuples(renumbered, false).mapEdges({ (pid, iter) =>
        val gen = new java.util.Random(pid)
        iter.map(e => LDA.combineTopics(gen.nextInt(nT), 0))
    })
    // Compute the topic histograms (factors) for each word and document
    val newCounts = gTmp.mapReduceTriplets[Counts](
      e => Iterator((e.srcId, makeCountsFromTopic(nT, e.attr)), (e.dstId, makeCountsFromTopic(nT, e.attr))),
      (a, b) => combineCounts(a,b)
    )
    // Update the graph with the factors
    gTmp.outerJoinVertices(newCounts)({(_, _, newFactorOpt) => makeHistogramFromCounts(newFactorOpt.get) }).cache()
  }

  /**
   * Get the word vertices by filtering on non-negative vertices
   * @return Word vertices
   */
  def wordVertices: VertexRDD[Histogram] = graph.vertices.filter{ case (vid, c) => vid >= 0 }

  /**
   * Get the document vertices by filtering on negative vertices
   * @return Document vertices
   */
  def docVertices: VertexRDD[Histogram] = graph.vertices.filter{ case (vid, c) => vid < 0 }

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
  var totalHistogram = makeHistogramFromCounts(graph.edges.map(e => e.attr)
    .aggregate(new Counts(nTopics))(LDA.combineTopicIntoCounts, LDA.combineCounts))

  /**
   * List to track time spent doing Gibbs sampling
   */
  val resampleTimes = new mutable.ListBuffer[Long]()

  /**
   * List to track time spent updating the counts on the graph
   */
  val updateCountsTimes = new mutable.ListBuffer[Long]()

  /**
   * List to track time spent updating the global topic histogram
   */
  val globalCountsTimes = new mutable.ListBuffer[Long]()

  /**
   * List to track negative log likelihood
   */
  val likelihoods = new mutable.ListBuffer[Double]()

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
  def train(iterations: Int) {
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

      val totalHistogramBroadcast = sc.broadcast(totalHistogram)

      // Shadowing because scala's closure capture would otherwise serialize the model object
      val a = alpha
      val b = beta
      val nt = nTopics
      val nw = nWords

      // Re-sample all the tokens
      var tempTimer: Long = System.nanoTime()
      val parts = graph.edges.partitions.size
      val interIter = internalIteration
      graph = graph.mapTriplets({(pid: PartitionID, iter: Iterator[EdgeTriplet[Histogram, Topic]]) =>
        val gen = new java.util.Random(parts * interIter + pid)
        iter.map({ token =>
          LDA.sampleToken(gen, token, totalHistogramBroadcast, nt, a, b, nw)
        })
      }, TripletFields.All)
      if (loggingTime && i % loggingInterval == 0) {
        graph.cache().triplets.count()
        resampleTimes += System.nanoTime() - tempTimer
      }

      // Update the counts
      tempTimer = System.nanoTime()

      //val newCounts = graph.triplets
      //                   .flatMap(e => {Iterator((e.srcId, e.attr), (e.dstId, e.attr))})
      //                   .aggregateByKey(new Array[Int](nt))(LDA.combineTopicIntoHistogram(_, _), LDA.combineHistograms(_, _))
      val deltas = graph.edges
        .flatMap(e => {
            val topic = e.attr
            val old = LDA.oldTopic(topic)
            val current = LDA.currentTopic(topic)
            var result: Iterator[(VertexId, Topic)] = Iterator.empty
            if (old != current) {
              result = Iterator((e.srcId, e.attr), (e.dstId, e.attr))
            }
            result
          })
        .aggregateByKey(new Array[Int](nt))(LDA.combineDeltaIntoCounts, LDA.combineCounts)

      graph = graph.outerJoinVertices(deltas)({(_, oldHistogram, vertexDeltasOption) =>
        if (vertexDeltasOption.isDefined) {
          val vertexDeltas = vertexDeltasOption.get
          val counts = (oldHistogram.counts, vertexDeltas).zipped.map(_ + _)
          makeHistogramFromCounts(counts)
        } else {
          makeHistogramFromCounts(oldHistogram.counts)
        }
      }).cache()

      //graph = graph.outerJoinVertices(newCounts)({(_, _, newFactorOpt) => newFactorOpt.get }).cache
      if (loggingTime && i % loggingInterval == 0) {
        graph.cache().triplets.count()
        updateCountsTimes += System.nanoTime() - tempTimer
      }

      // Recompute the global counts (the actual action)
      tempTimer = System.nanoTime()
      totalHistogram = makeHistogramFromCounts(graph.edges.map(e => e.attr)
        .aggregate(new Array[Int](nt))(LDA.combineTopicIntoCounts, LDA.combineCounts))
      assert(totalHistogram.counts.sum == nTokens)
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
          val tpl: (Int, WordId) = (factor.counts(t), wordId)
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
      totalHistogram.counts.map(v => Gamma.logGamma(v + nw * b)).sum +
      wordVertices.map({ case (id, histogram) => histogram.counts.map(v => Gamma.logGamma(v + b)).sum})
                  .reduce(_ + _)
    val logPZ =
      nd * (Gamma.logGamma(nt * a) - nt * logAlpha) +
      docVertices.map({ case (id, histogram) =>
        histogram.counts.map(v => Gamma.logGamma(v + a)).sum - Gamma.logGamma(nt * a + histogram.counts.sum)
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
