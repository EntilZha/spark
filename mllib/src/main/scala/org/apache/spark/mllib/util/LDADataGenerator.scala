
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
    val data:RDD[(LDA.WordId, LDA.DocId)] = sc.parallelize(0 until nDocs).flatMap({ docId =>
      // Initialize the per document dirichlet distribution identically for each document
      Random.seed(seed)
      val documentTopicDirichlet = new Dirichlet[DenseVector[Double], Int](DenseVector.fill[Double](nTopics, alpha))
      // Set the seed so each document has a unique and deterministic dirichlet sample for use in the categorical
      // Distribution to sample the word.
      Random.seed(seed + docId)
      val documentCategoricalDistribution = new Multinomial[DenseVector[Double], Int](documentTopicDirichlet.draw())
      (0 until nTokensPerDoc).map({ _ =>
        val topic = documentCategoricalDistribution.sample()
        val wtcdValue = wordTopicCategoricalDistributions.value
        val word:LDA.WordId = wtcdValue(topic).sample()
        val doc:LDA.DocId = docId
        (word, doc)
      })
    })
    return data
  }
}
