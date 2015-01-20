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

package org.apache.spark.mllib.topicmodeling

import breeze.linalg.DenseVector
import org.apache.spark.annotation.DeveloperApi
import org.apache.commons.math3.special.Gamma
import org.apache.spark.graphx._
import org.apache.spark.mllib.linalg.{Vector, Matrix}
import org.apache.spark.mllib.util.TimeTracker
import org.apache.spark.rdd.RDD
import org.apache.spark.util.BoundedPriorityQueue
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.util.Utils

import scala.collection.mutable

/**
 * :: DeveloperApi ::
 *
 * Latent Dirichlet Allocation (LDA), a topic model designed for text documents.
 *
 * Terminology:
 *  - "word" = "term": an element of the vocabulary
 *  - "token": instance of a term appearing in a document
 *  - "topic": multinomial distribution over words representing some concept
 *
 * Currently, the underlying implementation uses Expectation-Maximization (EM), implemented
 * according to the Asuncion et al. (2009) paper referenced below.
 *
 * References:
 *  - Original LDA paper (journal version):
 *    Blei, Ng, and Jordan.  "Latent Dirichlet Allocation."  JMLR, 2003.
 *     - This class implements their "smoothed" LDA model.
 *  - Paper which clearly explains several algorithms, including EM:
 *    Asuncion, Welling, Smyth, and Teh.
 *    "On Smoothing and Inference for Topic Models."  UAI, 2009.
 *
 * NOTE: This is currently marked DeveloperApi since it is under active development and may undergo
 *       API changes.
 */
@DeveloperApi
class LDA private (
                    private var k: Int,
                    private var maxIterations: Int,
                    private var topicSmoothing: Double,
                    private var termSmoothing: Double,
                    private var seed: Long,
                    private var algorithm: LDA.LearningAlgorithm.LearningAlgorithm) extends Logging {

  import LDA._


  def this() = this(k = 10, maxIterations = 20, topicSmoothing = -1, termSmoothing = -1,
    seed = Utils.random.nextLong(), algorithm = LDA.LearningAlgorithm.Gibbs)

  /**
   * Number of topics to infer.  I.e., the number of soft cluster centers.
   * (default = 10)
   */
  def getK: Int = k

  def setK(k: Int): this.type = {
    require(k > 0, s"LDA k (number of clusters) must be > 0, but was set to $k")
    this.k = k
    this
  }

  /**
   * Topic smoothing parameter (commonly named "alpha").
   *
   * This is the parameter to the Dirichlet prior placed on the per-document topic distributions
   * ("theta").  We use a symmetric Dirichlet prior.
   *
   * This value should be > 0.0, where larger values mean more smoothing (more regularization).
   * If set to -1, then topicSmoothing is set automatically.
   *  (default = -1 = automatic)
   *
   * Automatic setting of parameter:
   *  - For EM: default = (50 / k) + 1.
   *     - The 50/k is common in LDA libraries.
   *     - The +1 follows Asuncion et al. (2009), who recommend a +1 adjustment for EM.
   */
  def getTopicSmoothing: Double = topicSmoothing

  def setTopicSmoothing(topicSmoothing: Double): this.type = {
    require(topicSmoothing > 0.0 || topicSmoothing == -1.0,
      s"LDA topicSmoothing must be > 0 (or -1 for auto), but was set to $topicSmoothing")
    if (topicSmoothing > 0.0 && topicSmoothing <= 1.0) {
      logWarning(s"LDA.topicSmoothing was set to $topicSmoothing, but for EM, we recommend > 1.0")
    }
    this.topicSmoothing = topicSmoothing
    this
  }

  /**
   * Term smoothing parameter (commonly named "eta").
   *
   * This is the parameter to the Dirichlet prior placed on the per-topic word distributions
   * (which are called "beta" in the original LDA paper by Blei et al., but are called "phi" in many
   *  later papers such as Asuncion et al., 2009.)
   *
   * This value should be > 0.0.
   * If set to -1, then termSmoothing is set automatically.
   *  (default = -1 = automatic)
   *
   * Automatic setting of parameter:
   *  - For EM: default = 0.1 + 1.
   *     - The 0.1 gives a small amount of smoothing.
   *     - The +1 follows Asuncion et al. (2009), who recommend a +1 adjustment for EM.
   */
  def getTermSmoothing: Double = termSmoothing

  def setTermSmoothing(termSmoothing: Double): this.type = {
    require(termSmoothing > 0.0 || termSmoothing == -1.0,
      s"LDA termSmoothing must be > 0 (or -1 for auto), but was set to $termSmoothing")
    if (termSmoothing > 0.0 && termSmoothing <= 1.0) {
      logWarning(s"LDA.termSmoothing was set to $termSmoothing, but for EM, we recommend > 1.0")
    }
    this.termSmoothing = termSmoothing
    this
  }

  /**
   * Maximum number of iterations for learning.
   * (default = 20)
   */
  def getMaxIterations: Int = maxIterations

  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  /** Random seed */
  def getSeed: Long = seed

  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Learn an LDA model using the given dataset.
   *
   * @param documents  RDD of documents, where each document is represented as a vector of term
   *                   counts plus an ID.  Document IDs must be >= 0.
   * @return  Inferred LDA model
   */
  def runFromBagOfWords(documents: RDD[Document]): DistributedLDAModel = {
    val topicSmoothing = if (this.topicSmoothing > 0) {
      this.topicSmoothing
    } else {
      (50.0 / k) + 1.0
    }
    val termSmoothing = if (this.termSmoothing > 0) {
      this.termSmoothing
    } else {
      1.1
    }
    var state = algorithm match {
      case LearningAlgorithm.Gibbs => LDA.GibbsLearningState.initialStateFromBagOfWords(documents, k, topicSmoothing, termSmoothing, seed)
      case LearningAlgorithm.EM => LDA.GibbsLearningState.initialStateFromBagOfWords(documents, k, topicSmoothing, termSmoothing, seed)
    }
    var iter = 0
    while (iter < maxIterations) {
      state = state.next()
      iter += 1
    }
    new DistributedLDAModel(state)
  }

  def runFromEdges(edges: RDD[(WordId, DocId)]): DistributedLDAModel = {
    val topicSmoothing = if (this.topicSmoothing > 0) {
      this.topicSmoothing
    } else {
      (50.0 / k) + 1.0
    }
    val termSmoothing = if (this.termSmoothing > 0) {
      this.termSmoothing
    } else {
      1.1
    }
    var state = algorithm match {
      case LearningAlgorithm.Gibbs => LDA.GibbsLearningState.initialStateFromEdges(edges, k, topicSmoothing, termSmoothing, seed)
      case LearningAlgorithm.EM => LDA.GibbsLearningState.initialStateFromEdges(edges, k, topicSmoothing, termSmoothing, seed)
    }
    var iter = 0
    while (iter < maxIterations) {
      state = state.next()
      iter += 1
    }
    new DistributedLDAModel(state)
  }
}

/**
 * LDA contains utility methods used in the LDA class. These are mostly
 * methods which will be serialized during computation so cannot be methods.
 */
object LDA {
  object LearningAlgorithm extends Enumeration {
    type LearningAlgorithm = Value
    val Gibbs, EM = Value
  }
  type TopicCounts = Array[Int]
  case class Document(counts: Vector, id: Long)
  type DocId = Long
  type WordId = Long
  def isDocumentVertex(v: Long): Boolean = v < 0
  def isWordVertex(v: Long): Boolean = v >= 0
  trait LearningState {
    def logPrior(): Double
    def logLikelihood(): Double
    def collectTopicTotals(): TopicCounts
    def topicDistributions(): RDD[(Long, Vector)]
    def describeTopics(maxTermsPerTopic: Long): Array[Array[(Double, String)]]
    def topicsMatrix: Matrix
    def next(): LearningState
    def vocabSize(): Long
    def k: Int
  }

  trait LearningStateInitialization {
    def initialStateFromBagOfWords(docs: RDD[Document], k: Int, alpha: Double, beta: Double, seed: Long): LearningState
    def initialStateFromEdges(edges: RDD[(WordId, DocId)], k: Int, alpha: Double, beta: Double, seed: Long): LearningState
  }

  object GibbsLearningState extends LearningStateInitialization {
    import GibbsLearningState._

    /**
     * These two variables are used for turning indexing on and off when creating new histograms
     */
    val IndexTrue: DocId = -1L
    val IndexFalse: WordId = 1L
    /**
     * Topic Long is composed of two Ints representint topics.
     * The left Int represents the current topic, right Int represents prior topic.
     */
    type Topic = Long

    case class Histogram(counts: TopicCounts, index: Option[Index], normSum: Double) extends Serializable

    /**
     * Index holds two arrays which contain:
     * First: argsort of the counts from max to min
     * Second: for each topic, its position in the argsort
     */
    case class Index(argsort: Array[Int], lookup: Array[Int]) extends Serializable

    class Posterior(docs: VertexRDD[Histogram], words: VertexRDD[Histogram])


    def getCurrentTopic(topic: Topic): Int = {
      (topic >> 32).asInstanceOf[Int]
    }

    def getOldTopic(topic: Topic): Int = {
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
    def combineCounts(a: TopicCounts, b: TopicCounts): TopicCounts = {
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
    def combineTopicIntoCounts(a: TopicCounts, topic: Topic): TopicCounts = {
      a(getCurrentTopic(topic)) += 1
      a
    }

    def combineDeltaIntoCounts(a: TopicCounts, topic: Topic): TopicCounts = {
      a(getOldTopic(topic)) -= 1
      a(getCurrentTopic(topic)) += 1
      a
    }

    /**
     * Creates a new histogram then applies the deltas in place to maintain the
     * argsort order. It is assumed that deltas are sparse so that
     * the bubblesort used fast.
     * @param oldHistogram
     * @param deltas
     * @return
     */
    def applyDeltasToHistogram(oldHistogram: Histogram, deltas: Array[Int], parameter: Double): Histogram = {
      if (oldHistogram.index.isEmpty) {
        val counts = new TopicCounts(oldHistogram.counts.length)
        var i = 0
        while (i < counts.length) {
          counts(i) = oldHistogram.counts(i) + deltas(i)
          i += 1
        }
        makeHistogramFromCounts(counts, IndexFalse, parameter)
      } else {
        val index = Index(oldHistogram.index.get.argsort.clone(), oldHistogram.index.get.lookup.clone())
        var normSum = oldHistogram.normSum
        val counts = oldHistogram.counts.clone()
        for (k <- 0 until deltas.length) {
          if (deltas(k) != 0) {
            normSum = applyDeltaToHistogram(counts, index, normSum, k, deltas(k), parameter)
          }
        }
        Histogram(counts, Option(index), normSum)
      }
    }

    /**
     * Applies the deltas to the histogram, then returns the new normSum with the
     * topic changes.
     * @param counts
     * @param index
     * @param oldNormSum
     * @param topic
     * @param delta
     * @param parameter
     * @return
     */
    def applyDeltaToHistogram(counts: TopicCounts, index: Index, oldNormSum: Double, topic: Int, delta: Int, parameter: Double): Double = {
      val argsort = index.argsort
      val lookup = index.lookup
      var normSum = oldNormSum
      normSum -= math.pow(counts(topic) + parameter, 3)
      counts(topic) += delta
      normSum += math.pow(counts(topic) + parameter, 3)
      val c = counts(topic)
      var i = lookup(topic)
      if (delta > 0) {
        while (i > 0 && c > counts(argsort(i - 1))) {
          var tmp = argsort(i)
          argsort(i) = argsort(i - 1)
          argsort(i - 1) = tmp
          tmp = lookup(argsort(i))
          lookup(argsort(i)) = lookup(argsort(i - 1))
          lookup(argsort(i - 1)) = tmp
          i -= 1
        }
      } else if (delta < 0) {
        while (i < counts.length - 1 && c < counts(argsort(i + 1))) {
          var tmp = argsort(i)
          argsort(i) = argsort(i + 1)
          argsort(i + 1) = tmp
          tmp = lookup(argsort(i))
          lookup(argsort(i)) = lookup(argsort(i + 1))
          lookup(argsort(i + 1)) = tmp
          i += 1
        }
      }
      normSum
    }

    /**
     * Creates a factor with topic added to it.
     * @param nTopics Number of topics
     * @param topic Topic to start with
     * @return New factor with topic added to it
     */
    def makeCountsFromTopic(nTopics: Int, topic: Topic): TopicCounts = {
      val counts = new TopicCounts(nTopics)
      counts(getCurrentTopic(topic)) += 1
      counts
    }

    /**
     * Creates a new histogram from counts, counts becomes part of the new histogram
     * object.
     * @param counts
     * @param vid
     * @param parameter Parameter for computing norm sum, either alpha or beta
     * @return
     */
    def makeHistogramFromCounts(counts: TopicCounts, vid: VertexId, parameter: Double): Histogram = {
      if (vid < 0) {
        val argsort = breeze.linalg.argsort(new DenseVector(counts)).toArray.reverse
        val lookup = new Array[Int](counts.length)
        for (i <- 0 until lookup.length) {
          val topic = argsort(i)
          lookup(topic) = i
        }
        val index = Index(argsort, lookup)
        Histogram(counts, Option(index), computeNormSum(counts, parameter))
      } else {
        Histogram(counts, Option.empty, computeNormSum(counts, parameter))
      }
    }

    def makeEmptyHistogram(nTopics: Int): Histogram = {
      Histogram(new TopicCounts(nTopics), Option.empty, 0)
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
                              delimiter: String = " "): RDD[(WordId, DocId)] = {
      val sc = lines.sparkContext
      val numDocs = lines.count()
      val docIds: RDD[DocId] = sc.parallelize((0L until numDocs).toArray)
      val docsWithIds = lines.zip(docIds)
      val edges: RDD[(WordId, DocId)] = docsWithIds.flatMap { case (line: String, docId: DocId) =>
        val words = line.split(delimiter)
        val docEdges = words.map(word => (vocabLookup(word), docId))
        docEdges
      }
      edges
    }

    def computeNormSum(counts: TopicCounts, parameter: Double): Double = {
      var normSum: Double = 0
      if (parameter < 0) {
        return 0
      }
      var i = 0
      while (i < counts.length) {
        normSum += math.pow(counts(i) + parameter, 3)
        i += 1
      }
      normSum
    }

    /**
     * Re-samples the given token/triplet to a new topic
     * @param randomDouble Random number generator
     * @param totalHistogram roadcast Total histogram of topics
     * @param nt Number of topics
     * @param alpha Parameter for dirichlet prior on per document topic distributions
     * @param beta Parameter for the dirichlet prior on per topic word distributions
     * @param nw Number of words in corpus
     * @return New topic for token/triplet
     */
    def sampleToken(randomDouble: Double,
                    topic: Topic,
                    docHistogram: Histogram,
                    wordHistogram: Histogram,
                    totalHistogram: Histogram,
                    totalNorm: Double,
                    nt: Int,
                    alpha: Double,
                    beta: Double,
                    nw: Long): Topic = {
      val totalCounts = totalHistogram.counts
      val wHist = wordHistogram.counts
      val dHist = docHistogram.counts
      val oldTopic = getCurrentTopic(topic)
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
      val u = randomDouble * conditionalSum
      assert(u < conditionalSum)
      // Draw the new topic from the multinomial
      var newTopic = 0
      var cumsum = conditional(newTopic)
      while (cumsum < u) {
        newTopic += 1
        cumsum += conditional(newTopic)
      }
      combineTopics(newTopic, oldTopic)
    }

    def countWithoutTopic(counts: TopicCounts, index: Int, topic: Int): Int = {
      if (index == topic) {
        counts(index) - 1
      } else {
        counts(index)
      }
    }

    /**
     * This is a fast approximation to the square root used in the core Gibbs Sampling
     * step of LDA. Inspiration comes from acbrt1 from here:
     * http://www.hackersdelight.org/hdcodetxt/acbrt.c.txt
     * @param xVal
     * @return
     */
    def cubeRoot(xVal: Double): Double = {
      val x0: Float = xVal.toFloat
      var x: Float = x0
      var ix: Int = java.lang.Float.floatToRawIntBits(x)
      ix = ix / 4 + ix / 16
      ix += +ix / 16
      ix += ix / 256
      ix += 0x2a5137a0
      x = java.lang.Float.intBitsToFloat(ix)
      x = 0.33333333F * (2.0F * x + x0 / (x * x))
      x.toDouble
    }

    def fastSampleToken(randomDouble: Double,
                        topic: Topic,
                        docHistogram: Histogram,
                        wordHistogram: Histogram,
                        totalHistogram: Histogram,
                        totalNorm: Double,
                        nt: Int,
                        alpha: Double,
                        beta: Double,
                        nw: Long): Topic = {
      val currentTopic = getCurrentTopic(topic)
      val topicOrder = docHistogram.index.get.argsort
      val aCounts = docHistogram.counts
      val bCounts = wordHistogram.counts
      val cCounts = totalHistogram.counts
      var aSquareSum: Double = docHistogram.normSum
      var bSquareSum: Double = wordHistogram.normSum
      var cSquareSum: Double = totalNorm
      var bits: Long = 0
      aSquareSum -= math.pow(aCounts(currentTopic) + alpha, 3)
      aSquareSum += math.pow(cCounts(currentTopic) - 1 + alpha, 3)
      bSquareSum -= math.pow(bCounts(currentTopic) + beta, 3)
      bSquareSum += math.pow(bCounts(currentTopic) - 1 + beta, 3)
      cSquareSum -= math.pow(1 / (cCounts(currentTopic) + nw * beta), 3)
      cSquareSum += math.pow(1 / (cCounts(currentTopic) - 1 + nw * beta), 3)
      assert(cSquareSum >= 0)
      val zBound = new Array[Double](nt)
      val sumP = new Array[Double](nt + 1)
      var u = randomDouble
      var i = 0
      var k = 0
      var a: Double = 0
      var b: Double = 0
      var c: Double = 0
      var t = 0
      var offset: Int = 0
      while (i < nt) {
        k = topicOrder(i)
        offset = (k == currentTopic).compare(true)
        a = aCounts(k) - offset + alpha
        b = bCounts(k) - offset + beta
        c = 1D / (cCounts(k) - offset + beta * nw)
        sumP(i + 1) = sumP(i)
        sumP(i + 1) += a * b * c
        bits = java.lang.Double.doubleToRawLongBits(aSquareSum - a * a * a)
        aSquareSum = java.lang.Double.longBitsToDouble(~(bits >> 63) & bits)
        bits = java.lang.Double.doubleToRawLongBits(bSquareSum - b * b * b)
        bSquareSum = java.lang.Double.longBitsToDouble(~(bits >> 63) & bits)
        bits = java.lang.Double.doubleToRawLongBits(cSquareSum - c * c * c)
        cSquareSum = java.lang.Double.longBitsToDouble(~(bits >> 63) & bits)
        zBound(i) = sumP(i + 1) + cubeRoot(aSquareSum * bSquareSum * cSquareSum)
        if (sumP(i + 1) >= u * zBound(i)) {
          if (i == 0 || u * zBound(i) > sumP(i)) {
            return combineTopics(k, currentTopic)
          } else {
            u = (u * zBound(i - 1) - sumP(i)) * zBound(i) / (zBound(i - 1) - zBound(i))
            t = 0
            while (t < i) {
              if (sumP(t + 1) >= u) {
                return combineTopics(topicOrder(t), currentTopic)
              }
              t += 1
            }
          }
        }
        i += 1
      }
      throw new Exception("fast sample token failed")
    }

    def vertices(edges: RDD[Edge[Topic]],
                 getVertex: Edge[Topic] => VertexId,
                 nTopics: Int,
                 vertexType: VertexId,
                 a: Double,
                 b: Double): RDD[(Long, Histogram)] = {
      edges.groupBy(getVertex(_)).aggregateByKey(makeEmptyHistogram(nTopics))({ case (histogram, iterator) =>
        iterator.foreach({ e =>
          histogram.counts(getCurrentTopic(e.attr)) += 1
        })
        val parameter = if (vertexType < 0) a else b
        makeHistogramFromCounts(histogram.counts, vertexType, parameter)
      }, { case (l, r) =>
        combineCounts(l.counts, r.counts)
        l
      })
    }
    def isArgSorted(arr: Array[Int], order: Array[Int]): Boolean = {
      for (i <- 1 until arr.length) {
        if (arr(order(i)) > arr(order(i - 1))) {
          return false
        }
      }
      true
    }

    // TODO
    def initialStateFromBagOfWords(docs: RDD[Document], k: Int, alpha: Double, beta: Double, seed: Long): GibbsLearningState = {
      throw new NotImplementedError()
    }
    def initialStateFromEdges(edges: RDD[(WordId, DocId)], k: Int, alpha: Double, beta: Double, seed: Long): GibbsLearningState = {
      val state = new GibbsLearningState(edges, k, alpha, beta)
      state.setup()
      state
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
  private[topicmodeling] class GibbsLearningState(@transient val tokens: RDD[(WordId, DocId)],
                      val nTopics: Int = 100,
                      val alpha: Double = 0.01,
                      val beta: Double = 0.01,
                      val loggingInterval: Int = 0,
                      val loggingLikelihood: Boolean = false,
                      val loggingTime: Boolean = false) extends LearningState with Serializable with Logging {
    import GibbsLearningState._
    var timer: TimeTracker = null
    private var sc: SparkContext = null
    private var graph: Graph[Histogram, Topic] = null
    var nWords: Long = 0
    var nDocs: Long = 0
    var nTokens: Long = 0
    var totalHistogram: Histogram = null
    var resampleTimes: mutable.ListBuffer[Long] = null
    var updateCountsTimes: mutable.ListBuffer[Long] = null
    var globalCountsTimes:mutable.ListBuffer[Long] = null
    var likelihoods: mutable.ListBuffer[Double] = null
    private var iteration = 0
    var modelIsSetup: Boolean = false

    def topicDistributions(): RDD[(Long, Vector)] = throw new NotImplementedError()

    def topicsMatrix: Matrix = throw new NotImplementedError()

    def describeTopics(maxTermsPerTopic: Long): Array[Array[(Double, String)]] = throw new NotImplementedError()

    def collectTopicTotals(): TopicCounts = throw new NotImplementedError()

    def logPrior(): Double = throw new NotImplementedError()

    def k: Int = nTopics

    def vocabSize(): Long = {
      nWords
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

    def setup(): Unit = {
      timer = new TimeTracker()
      resampleTimes = new mutable.ListBuffer[Long]()
      updateCountsTimes = new mutable.ListBuffer[Long]()
      globalCountsTimes = new mutable.ListBuffer[Long]()
      likelihoods = new mutable.ListBuffer[Double]()
      timer.start("setup")
      logInfo("Starting LDA setup")
      sc = tokens.sparkContext
      /**
       * The bipartite terms by document graph.
       */
      graph = {
        val nT = nTopics
        val a = alpha
        val b = beta
        // To setup a bipartite graph it is necessary to ensure that the document and
        // word ids are in a different namespace
        val edges: RDD[Edge[Topic]] = tokens.mapPartitionsWithIndex({ case (pid, iterator) =>
          val gen = new java.util.Random(pid)
          iterator.map({ case (wordId, docId) =>
            assert(wordId >= 0)
            assert(docId >= 0)
            val newDocId: DocId = -(docId + 1L)
            Edge(wordId, newDocId, combineTopics(gen.nextInt(nT), 0))
          })
        })
        val setupWordVertices = vertices(edges, _.srcId, nT, IndexFalse, a, b)
        val setupDocVertices = vertices(edges, _.dstId, nT, IndexTrue, a, b)
        Graph(setupDocVertices ++ setupWordVertices, edges).partitionBy(PartitionStrategy.EdgePartition1D)
      }.cache()
      nWords = wordVertices.count()
      nDocs = docVertices.count()
      nTokens = graph.edges.count()
      /**
       * The total counts for each topic
       */
      val counts = graph.edges.map(e => e.attr).aggregate(new TopicCounts(nTopics))(combineTopicIntoCounts, combineCounts)
      totalHistogram = makeHistogramFromCounts(graph.edges.map(e => e.attr)
        .aggregate(new TopicCounts(nTopics))(combineTopicIntoCounts, combineCounts), IndexFalse, -1)
      logInfo("LDA setup finished")
      timer.stop("setup")
      modelIsSetup = true
    }

    def next(): GibbsLearningState = {
      // Log the negative log likelihood
      if (loggingLikelihood && iteration % loggingInterval == 0) {
        val likelihood = logLikelihood()
        likelihoods += likelihood
      }
      // Shadowing because scala's closure capture would otherwise serialize the model object
      val a = alpha
      val b = beta
      val nt = nTopics
      val nw = nWords

      // Broadcast the topic histogram
      val totalHistogramBroadcast = sc.broadcast(totalHistogram)
      val totalNormSum = totalHistogram.counts.map(c => math.pow(1D / (c + nw * beta), 3)).sum
      val totalNormSumBroadcast = sc.broadcast(totalNormSum)

      // Re-sample all the tokens
      var tempTimer: Long = System.nanoTime()
      val parts = graph.edges.partitions.size
      val interIter = iteration
      graph = graph.mapTriplets({(pid: PartitionID, iter: Iterator[EdgeTriplet[Histogram, Topic]]) =>
        val gen = new java.util.Random(parts * interIter + pid)
        iter.map({ token =>
          val u = gen.nextDouble()
          if (iteration > 0) {
            fastSampleToken(u, token.attr, token.dstAttr, token.srcAttr, totalHistogramBroadcast.value, totalNormSumBroadcast.value, nt, a, b, nw)
          } else {
            sampleToken(u, token.attr, token.dstAttr, token.srcAttr, totalHistogramBroadcast.value, totalNormSumBroadcast.value, nt, a, b, nw)
          }
        })
      }, TripletFields.All)
      if (loggingTime && iteration % loggingInterval == 0 && iteration > 0) {
        graph.cache().triplets.count()
        resampleTimes += System.nanoTime() - tempTimer
      }

      // Update the counts
      tempTimer = System.nanoTime()

      val deltas = graph.edges
        .flatMap(e => {
        val topic = e.attr
        val old = getOldTopic(topic)
        val current = getCurrentTopic(topic)
        var result: Iterator[(VertexId, Topic)] = Iterator.empty
        if (old != current) {
          result = Iterator((e.srcId, e.attr), (e.dstId, e.attr))
        }
        result
      })
        .aggregateByKey(new Array[Int](nt))(combineDeltaIntoCounts, combineCounts)

      graph = graph.outerJoinVertices(deltas)({(vid, oldHistogram, vertexDeltasOption) =>
        if (vertexDeltasOption.isDefined) {
          val vertexDeltas = vertexDeltasOption.get
          val parameter = if (vid < 0) a else b
          val histogram = applyDeltasToHistogram(oldHistogram, vertexDeltas, parameter)
          histogram
        } else {
          Histogram(oldHistogram.counts, oldHistogram.index, oldHistogram.normSum)
        }
      }).cache()

      if (loggingTime && iteration % loggingInterval == 0 && iteration > 0) {
        graph.cache().triplets.count()
        updateCountsTimes += System.nanoTime() - tempTimer
      }

      // Recompute the global counts (the actual action)
      tempTimer = System.nanoTime()
      totalHistogram = makeHistogramFromCounts(graph.edges.map(e => e.attr)
        .aggregate(new Array[Int](nt))(combineTopicIntoCounts, combineCounts), IndexFalse, -1)
      assert(totalHistogram.counts.sum == nTokens)
      if (loggingTime && iteration % loggingInterval == 0 && iteration > 0) {
        globalCountsTimes += System.nanoTime() - tempTimer
      }

      iteration += 1
      this
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
      new Posterior(words, docs)
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
      logInfo(s"Topics: $nTopics")
      logInfo(s"Alpha: $alpha")
      logInfo(s"Beta: $beta")
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
}

