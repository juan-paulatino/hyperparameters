def objective(params):
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from hyperopt import STATUS_OK

    data_df = get_training_data() # This is just an example!
    splits = data_df.randomSplit([0.7, 0.3])
    training_df = splits[0]
    validation_df = splits[1]

    # Train a model using the provided hyperparameter values
    lr = LogisticRegression(labelCol="label", featuresCol="features",
                            maxIter=params['Iterations'],
                            regParam=params['Regularization'])
    model = lr.fit(training_df)

    # Evaluate the model
    predictions = model.transform(validation_df)
    eval = MulticlassClassificationEvaluator(labelCol="label",
                                             predictionCol="prediction",
                                             metricName="accuracy")
    accuracy = eval.evaluate(predictions)
    
    # Hyperopt *minimizes* the function, so return *negative* accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}