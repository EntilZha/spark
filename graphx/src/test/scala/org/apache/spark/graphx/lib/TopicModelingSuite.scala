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
import org.apache.spark.graphx.algorithms.LDA.{TopicId, DocId, WordId, Count, Factor}
import org.scalatest.{FunSuite, Matchers}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.graphx._
import org.apache.spark.rdd._

class TopicModelingSuite extends FunSuite with LocalSparkContext with Matchers {
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
      vocab.length should be (12)
      vocabLookup.contains("computer") should be (true)
      vocabLookup.contains("user") should be (true)
      vocabLookup.contains("trees") should be (true)
      vocabLookup.contains("time") should be (true)
      vocabLookup.contains("system") should be (true)
      vocabLookup.contains("survey") should be (true)
      vocabLookup.contains("response") should be (true)
      vocabLookup.contains("minors") should be (true)
      vocabLookup.contains("interface") should be (true)
      vocabLookup.contains("human") should be (true)
      vocabLookup.contains("graph") should be (true)
      vocabLookup.contains("eps") should be (true)
    }
  }
  test("Construct a factor from number of topics and a new topic") {
    val topicId0:TopicId = 0
    val topicId1:TopicId = 1
    val nTopics = 5
    val f = new Factor(nTopics)
    f(0) = 1
    f(1) = 0
    LDA.makeFactor(nTopics, topicId0) should equal (f)
    f(0) = 0
    f(1) = 1
    LDA.makeFactor(nTopics, topicId1) should equal (f)
  }
  test("Add topic to Factor") {
    val f = new Factor(2)
    f(0) = 1
    f(1) = 5
    val result = LDA.addEq(f, 0)
    val fExpect = new Factor(2)
    fExpect(0) = 2
    fExpect(1) = 5
    f should equal (fExpect)
    result should equal (fExpect)
  }
  test("Add two factors together") {
    val f1 = new Factor(2)
    val f2 = new Factor(2)
    f1(0) = 2
    f1(1) = 4
    f2(0) = 5
    f2(1) = 2
    val result = LDA.addEq(f1, f2)
    val fExpect = new Factor(2)
    fExpect(0) = 7
    fExpect(1) = 6
    result should be (fExpect)
    f1 should be (fExpect)
  }
}
