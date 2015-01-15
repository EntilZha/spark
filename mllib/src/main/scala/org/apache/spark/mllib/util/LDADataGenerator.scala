
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

package org.apache.spark.mllib.util

import breeze.linalg.DenseVector
import breeze.stats.distributions.{Multinomial, Dirichlet}
import org.apache.spark.SparkContext
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.graphx.lib.LDA
import org.apache.spark.rdd.RDD
import breeze.stats.distributions
import org.jblas.util.Random


/**
 * :: DeveloperApi ::
 * Generate sample data used for topic modeling using Latent Dirichlet Allocation
 * Data is generated using the generative model for LDA described below

  $\alpha=$ Parameter for Dirichlet prior on the per-document topic distributions\\
  $\beta=$ Parameter for Dirichlet prior on the per-topic word distribution\\
  $\theta_i=$ Topic distribution for document i\\
  $\phi_k=$ Word distribution for topic k\\
  $z_{ij}=$ Topic for the jth word in document i\\
  $w_{ij}=$ Specific word for the jth word in document i\\

  The generative process behind is that documents are represented as random mixtures
  over latent topics, where each topic is characterized by a distribution over words. LDA assumes
  the following generative process for a corpus D consisting of M documents of length $N_i$\\

  1. Choose $\theta_i\sim Dir(\alpha)$ where $i\in\{1,...,M\}$ and $Dir(\alpha)$ is the Dirichlet distribution for parameter $\alpha$\\
  2. Choose $\phi_k\sim Dir(\beta)$, where $k\in\{1,...,K\}$\\
  3. For each of the word positions $i$,$j$, where $j\in\{1,...,N_i\}$, and $i\in\{1,...,M\}$
  \indent (a) Choose topic $z_{i,j}\sim Multinomial(\theta_i)$\\
  \indent (b) Choose a word $w_{i,j}\sim Multinomial(\phi_{i,j})$
 */
@DeveloperApi
object LDADataGenerator {
  def generateCorpus(
      sc:SparkContext,
      alpha:Double,
      beta:Double,
      nTopics:Int,
      nDocs:Int,
      nWords:Int,
      nTokensPerDoc:Int):RDD[(LDA.WordId, LDA.DocId)] = {
    val seed = 42
    Random.seed(seed)
    // Create the global dirichlet and categorical distributions over topics then broadcast them.
    val wordTopicDirichlet = new Dirichlet[DenseVector[Double], Int](DenseVector.fill[Double](nWords, beta))
    val wordTopicCategoricalDistributions = sc.broadcast((0 until nTopics).map({d =>
      new Multinomial[DenseVector[Double], Int](wordTopicDirichlet.draw())
    }))
    val perDocumentTokens:RDD[Seq[(LDA.WordId, LDA.DocId)]] = sc.parallelize(0 until nDocs).mapPartitions({docIds =>
      val dirichletVector = DenseVector.fill[Double](nTopics, alpha)
      val documentTopicDirichlet = new Dirichlet[DenseVector[Double], Int](dirichletVector)
      val documentTokens = docIds.map({docId =>
        // Set the seed so each document has a unique and deterministic dirichlet sample for use in the categorical
        // Distribution to sample the word.
        Random.seed(seed + docId)
        val documentCategoricalDistribution = new Multinomial[DenseVector[Double], Int](documentTopicDirichlet.draw())
        val tokens: Seq[(LDA.WordId, LDA.DocId)] = (0 until nTokensPerDoc).map({ _ =>
          val topic = documentCategoricalDistribution.sample()
          val wtcdValue = wordTopicCategoricalDistributions.value
          val word:LDA.WordId = wtcdValue(topic).sample()
          val doc:LDA.DocId = docId
          (word, doc)
        })
        tokens
      })
      documentTokens
    })
    val data:RDD[(LDA.WordId, LDA.DocId)] = perDocumentTokens.flatMap(doc => doc)
    data
  }

  def main(args: Array[String]): Unit = {
    if (args.length < 8) {
      println("Usage: LDADataGenerator " +
        "<master> <output_dir> <alpha> <beta> <nTopics> <nDocs> <nWords> <nTokensPerDoc>"
      )
      System.exit(1)
    }

    val sparkMaster:String = args(0)
    val outputPath:String = args(1)
    val alpha:Double = args(2).toDouble
    val beta:Double = args(3).toDouble
    val nTopics:Int = args(4).toInt
    val nDocs:Int = args(5).toInt
    val nWords:Int = args(6).toInt
    val nTokensPerDoc:Int = args(7).toInt
    val sc = new SparkContext(sparkMaster, "LDADataGenerator")
    val data = generateCorpus(sc, alpha, beta, nTopics, nDocs, nWords, nTokensPerDoc)
    data.saveAsTextFile(outputPath)
    sc.stop()
  }
}
