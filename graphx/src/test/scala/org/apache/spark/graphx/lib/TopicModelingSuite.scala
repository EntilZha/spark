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

import org.apache.spark.SparkContext._
import org.apache.spark.graphx._
import org.apache.spark.graphx.lib.LDA.{DocId, Factor, TopicId, WordId}
import org.apache.spark.rdd._
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class TopicModelingSuite extends FunSuite with LocalSparkContext with Matchers {
  test("Test edge generation") {
    withSpark { sc =>
      val lines = sc.textFile("data/lda-mini-test.txt")
      val tokens = lines.flatMap(line => line.split(" "))
      val (vocab:Array[String], vocabLookup:mutable.Map[String, WordId]) = LDA.extractVocab(tokens)
      // Extract the edges then reverse the wordId and docId to let us use groupByKey
      val edges:RDD[(LDA.WordId, LDA.DocId)] = LDA.edgesFromTextDocLines(lines, vocab, vocabLookup)
        .map{ case (wordId:WordId, docId:DocId) =>
        (docId, wordId)
      }
      // Reconstruct the documents to test the LDA.edgesFromTextDocLines call
      val docs:Array[String] = edges.groupByKey().collect().map{ case (docId:DocId, tokens:Iterable[WordId]) =>
        tokens.toArray.sorted.mkString
      }.sorted
      // Construct equivalent structure directly
      val docsExpect:Array[String] = lines.map({ line =>
        line.split(" ").map(token => vocabLookup(token)).sorted.mkString
      }).collect().sorted
      // Test that the arrays are equal. Do this by sorting each token list, converting to string representation
      docs should equal (docsExpect)
    }
  }
  test("Run lda-mini-test") {
    withSpark { sc =>
      val lines = sc.textFile("data/lda-mini-test.txt")
      val tokens = lines.flatMap(line => line.split(" "))
      val (vocab, vocabLookup) = LDA.extractVocab(tokens)
      val edges = LDA.edgesFromTextDocLines(lines, vocab, vocabLookup)
      val model = new LDA(edges, nTopics=2)
      model.train(20)
      val topWords = model.topWords(2)
      val t1w1 = vocab(topWords(0)(0)._2.toInt)
      val t1w2 = vocab(topWords(0)(1)._2.toInt)
      val t2w1 = vocab(topWords(1)(0)._2.toInt)
      val t2w2 = vocab(topWords(1)(1)._2.toInt)
      val t1words = Array(t1w1, t1w2).sorted.mkString(" ")
      val t2words = Array(t2w1, t2w2).sorted.mkString(" ")
      val wordsExpect1 = Array("system", "user").mkString(" ")
      val wordsExpect2 = Array("graph", "trees").mkString(" ")
      val t1check = t1words == wordsExpect1 && t1words != wordsExpect2 ||
        t1words == wordsExpect2 && t1words != wordsExpect1
      val t2check = t2words == wordsExpect1 && t2words != wordsExpect2 ||
        t2words == wordsExpect2 && t2words != wordsExpect1
      t1check should be (true)
      t2check should be (true)
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
