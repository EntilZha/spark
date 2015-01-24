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

package org.apache.spark.mllib.clustering

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.{Vectors, Matrices}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._

class GMMExpectationMaximizationSuite extends FunSuite with MLlibTestSparkContext {
  test("single cluster") {
    val data = sc.parallelize(Array(
      Vectors.dense(6.0, 9.0),
      Vectors.dense(5.0, 10.0),
      Vectors.dense(4.0, 11.0)
    ))
    
    // expectations
    val Ew = 1.0
    val Emu = Vectors.dense(5.0, 10.0)
    val Esigma = Matrices.dense(2, 2, Array(2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0))

    val seeds = Array(314589, 29032897, 50181, 494821, 4660)
    seeds.foreach { seed =>
      val gmm = new GaussianMixtureEM().setK(1).setSeed(seed).run(data)
      assert(gmm.weights(0) ~== Ew absTol 1E-5)
      assert(gmm.gaussians(0).mu ~== Emu absTol 1E-5)
      assert(gmm.gaussians(0).sigma ~== Esigma absTol 1E-5)
    }
  }
  
  test("two clusters") {
    val data = sc.parallelize(Array(
      Vectors.dense(-5.1971), Vectors.dense(-2.5359), Vectors.dense(-3.8220),
      Vectors.dense(-5.2211), Vectors.dense(-5.0602), Vectors.dense( 4.7118),
      Vectors.dense( 6.8989), Vectors.dense( 3.4592), Vectors.dense( 4.6322),
      Vectors.dense( 5.7048), Vectors.dense( 4.6567), Vectors.dense( 5.5026),
      Vectors.dense( 4.5605), Vectors.dense( 5.2043), Vectors.dense( 6.2734)
    ))
  
    // we set an initial gaussian to induce expected results
    val initialGmm = new GaussianMixtureModel(
      Array(0.5, 0.5),
      Array(
        new MultivariateGaussian(Vectors.dense(-1.0), Matrices.dense(1, 1, Array(1.0))),
        new MultivariateGaussian(Vectors.dense(1.0), Matrices.dense(1, 1, Array(1.0)))
      )
    )
    
    val Ew = Array(1.0 / 3.0, 2.0 / 3.0)
    val Emu = Array(Vectors.dense(-4.3673), Vectors.dense(5.1604))
    val Esigma = Array(Matrices.dense(1, 1, Array(1.1098)), Matrices.dense(1, 1, Array(0.86644)))
    
    val gmm = new GaussianMixtureEM()
      .setK(2)
      .setInitialModel(initialGmm)
      .run(data)
      
    assert(gmm.weights(0) ~== Ew(0) absTol 1E-3)
    assert(gmm.weights(1) ~== Ew(1) absTol 1E-3)
    assert(gmm.gaussians(0).mu ~== Emu(0) absTol 1E-3)
    assert(gmm.gaussians(1).mu ~== Emu(1) absTol 1E-3)
    assert(gmm.gaussians(0).sigma ~== Esigma(0) absTol 1E-3)
    assert(gmm.gaussians(1).sigma ~== Esigma(1) absTol 1E-3)
  }
}
