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

import org.apache.spark.graphx.algorithms.LDA
import org.scalatest.FunSuite

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.graphx._
import org.apache.spark.rdd._

class TopicModelingSuite extends FunSuite with LocalSparkContext {
  test("Simple LDA test with lda-minitest.txt") {
    withSpark { sc =>
      val data = sc.textFile("data/lda-mini-test.txt")
      val (vocab, vocabLookup) = LDA.extractVocab(data.flatMap(line => line.split(" ")))
    }
  }
  test("Vocabulary extraction") {
    withSpark { sc =>
      val data = sc.textFile("data/lda-mini-test.txt")
      val (vocab, vocabLookup) = LDA.extractVocab(data.flatMap(line => line.split(" ")))
      assert(vocab.length == 12)
      assert(vocabLookup.contains("computer"))
      assert(vocabLookup.contains("user"))
      assert(vocabLookup.contains("trees"))
      assert(vocabLookup.contains("time"))
      assert(vocabLookup.contains("system"))
      assert(vocabLookup.contains("survey"))
      assert(vocabLookup.contains("response"))
      assert(vocabLookup.contains("minors"))
      assert(vocabLookup.contains("interface"))
      assert(vocabLookup.contains("human"))
      assert(vocabLookup.contains("graph"))
      assert(vocabLookup.contains("eps"))
    }
  }
}
