const tf = require('@tensorflow/tfjs'); // No tfjs-node again
const { round } = require('lodash');
// const _ = require('lodash');

const defaultOptions = { learningRate: 0.1, iterations: 1000, batchSize: 10, decisionBoundary: 0.5 }; //'iterations' to disallow training to keep running forever | 'batchSize' for batch and stochastic GD

class LogisticRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        // console.log("Features: ", this.features.print());
        this.labels = tf.tensor(labels);


        this.crossEntropyHistory = []; // error factor for classification
        // this.cHistory = [];
        // this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1); // Pad features to match its shape with weights
        // console.log("Features new: ", this.features.print());
        this.options = Object.assign({ ...defaultOptions }, options);

        this.weights = tf.zeros([this.features.shape[1], 1]); // Initial guesses for c / c1, c2 .. cN and m and future proofing for more features
    }

    processFeatures(features) {
        features = tf.tensor(features);


        // Standardization MUST use same mean and variance as training data

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5))
        } else features = this.standardize(features);

        features = tf.ones([features.shape[0], 1]).concat(features, 1); // The ones do NOT need to be standardized -- TF may give different output based on handlers

        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }

    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize); // Decide how many times to run Batch GD and how many batches there are
        // ex. for 300 observations and 10 batch size, batch quantiy will be 30
        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) // for each iteration, each batch will be run <batchQuantity> times
            {

                const { batchSize } = this.options;
                const startIndex = j * batchSize;
                const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]); // for first iteration, startIndex = 0
                const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
                this.gradientDescent(featureSlice, labelSlice);
            }
            // Example: Batch qty: 30, batch size: 10
            /* j=0 ; 0 < 30
            startIndex = 0 * 10 = 0
            featureSlice([0,0], [10, all columns])

            j=1; 1<30
            startIndex = 1 * 10 = 10;
            featureSlice([10,0], [10,-1]) // second arguments 10 means 10 more rows
            
            */

            // this.cHistory.push(this.weights.get(0,0)); // get value of C based on weights

            this.recordCrossEntropy(); // record errors in training phase
            this.updateLearningRate();
        }
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        console.log(predictions.shape)
        testLabels = tf.tensor(testLabels);

        const incorrect = predictions.sub(testLabels).abs().sum().get();
        console.log(incorrect)
        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).sigmoid(); //sigmoid ( mx + c )-- this.features [  <1 col for padding> , <columns for features>] and this.weights [ <value of cSlope> , <value(s) of mSlope for one of more features >]
        const differences = currentGuesses.sub(labels);

        const slopes = features.transpose().matMul(differences).div(features.shape[0]); // calculate slope wrt m and c -- Transpose is to match shape of differences -- div does element wise divison

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate))

    }

    recordCrossEntropy() { // created for modifying learning rate based on new Cross Entropy

        const guesses = this.features.matMul(this.weights).sigmoid();

        const termOne = this.labels.transpose().matMul(guesses.log());

        const termTwo = this.labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).log()); // We did -Actual + 1

        const total = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);

        this.crossEntropyHistory.unshift(total);



    }

    updateLearningRate() {
        if (this.crossEntropyHistory.length < 2) // enough records to update
            return;

        if (this.crossEntropyHistory[0] > this.crossEntropyHistory[1]) //MSE increased, bad learning rate
            this.options.learningRate /= 2;
        else this.options.learningRate *= 1.05;
    }

    predict(observations) { // observations will HAVE to be in the order of the feature columns - horsepower first.. and so on. -> y= mx+c
        return this.processFeatures(observations).matMul(this.weights).sigmoid().greater(this.options.decisionBoundary).cast('float32'); // If greater, tensor value is replaced by 1 else it is replaced by 0 BUT THIS IS BOOLEAN IN REALITY which is why we cast

    }

}

module.exports = LogisticRegression;