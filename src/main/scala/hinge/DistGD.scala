package distopt.solvers

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import distopt.utils._
import breeze.linalg.{Vector, NumericOps, DenseVector, SparseVector}


object DistGD {

  /**
   * Implementation of distributed subgradient descent
   * Using hinge-loss SVM objective.
   * 
   * @param data RDD of all data examples
   * @param params Algorithmic parameters
   * @param debug Systems/debugging parameters
   * @return
   */
  def runDistGD(
    data: RDD[LabeledPoint],
    params: Params,
    debug: DebugParams) : Vector[Double] = {
    
    var dataArr = data.mapPartitions(x => Iterator(x.toArray))
    val parts = data.partitions.size 	// number of partitions of the data, K in the paper
    println("\nRunning DistGD on "+params.n+" data examples, distributed over "+parts+" workers")
    
    // initialize w
    var w = params.wInit.copy
    
    for(t <- 1 to params.numRounds){

      // update step size
      val step = 1 / (params.beta * (t))

      // find updates to w
      val updates = dataArr.mapPartitions(partitionUpdate(_, w, params.lambda, ((t-1) * params.localIters * parts), params.localIters, parts, debug.seed + t), preservesPartitioning = true).persist()
      val primalUpdates = updates.reduce(_ + _)
      val updateNorm = primalUpdates.norm(2)
      w += (primalUpdates * (step / updateNorm))

      // optionally calculate errors
      if (debug.debugIter > 0 && t % debug.debugIter == 0) {
        println("Iteration: " + t)
        println("primal objective: " + OptUtils.computePrimalObjective(data, w, params.lambda))
        if (debug.testData != null) { println("test error: " + OptUtils.computeClassificationError(debug.testData, w)) }
      }
    }

    return w
  }


  /**
   * Performs one round of local updates using SGD steps on the local points, 
   * Will perform localIters many updates per worker.
   * 
   * @param localData
   * @param wInit
   * @param lambda
   * @param t
   * @param parts
   * @param seed
   * @return
   */
  def partitionUpdate(
    localData: Iterator[Array[LabeledPoint]], 
    wInit: Vector[Double], 
    lambda:Double, 
    t:Double, 
    localIters:Int, 
    parts:Int,
    seed: Int) : Iterator[Vector[Double]] = {

    val dataArr = localData.next()
    val nLocal = dataArr.length
    var w = wInit.copy
    var deltaW = Vector.zeros[Double](wInit.length)

    // perform updates
    for (idx <- 0 to nLocal) {

      // randomly select an element
      val currPt = dataArr(idx)
      var y = currPt.label
      val x = currPt.features

      // calculate stochastic sub-gradient (here for SVM hinge loss)
      val eval = 1.0 - (y * (x.dot(w)))

      // stochastic sub-gradient, update
      if (eval > 0) {
        val update = x * y
        deltaW += update
      }
    }
    deltaW -= lambda*wInit;

    // return change in weight vector
    return Iterator(deltaW)
  }

}
