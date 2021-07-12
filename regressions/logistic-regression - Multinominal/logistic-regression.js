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

        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]); // Initial guesses for c / c1, c2 .. cN and multiple classification labels and future proofing for more features
    }

    processFeatures(features) {
        features = tf.tensor(features);

        // debugger;
        // Standardization MUST use same mean and variance as training data

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5))
        } else features = this.standardize(features);

        features = tf.ones([features.shape[0], 1]).concat(features, 1); // The ones do NOT need to be standardized -- TF may give different output based on handlers

        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        const filler = variance.cast('bool').logicalNot().cast('float32'); // fixing divide by 0

        this.mean = mean;
        this.variance = variance.add(filler);

        return features.sub(mean).div(this.variance.pow(0.5));
    }

    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize); // Decide how many times to run Batch GD and how many batches there are
        // ex. for 300 observations and 10 batch size, batch quantiy will be 30
        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) // for each iteration, each batch will be run <batchQuantity> times
            {

                const { batchSize } = this.options;
                const startIndex = j * batchSize;

                this.weights = tf.tidy(() => { // To remove weakmap references of these tensors as soon as we're done using them

                    const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]); // for first iteration, startIndex = 0
                    const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
                    return this.gradientDescent(featureSlice, labelSlice);
                });
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
        // console.log(predictions.shape)
        testLabels = tf.tensor(testLabels).argMax(1); // Get highest values along row (1)

        const incorrect = predictions.notEqual(testLabels).sum().get(); // abs is not needed 
        // console.log(incorrect)
        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).softmax(); //softmax ( mx + c )-- this.features [  <1 col for padding> , <columns for features>] and this.weights [ <value of cSlope> , <value(s) of mSlope for one of more features >]
        const differences = currentGuesses.sub(labels);

        const slopes = features.transpose().matMul(differences).div(features.shape[0]); // calculate slope wrt m and c -- Transpose is to match shape of differences -- div does element wise divison

        return this.weights.sub(slopes.mul(this.options.learningRate))

    }

    recordCrossEntropy() { // created for modifying learning rate based on new Cross Entropy

        const cost = tf.tidy(() => {
            // debugger;

            //We add the negligible 10^-7 power to prevent taking log of 0
            const guesses = this.features.matMul(this.weights).sigmoid();

            const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log()); // log of 0 is -Infinity and of a -ve number does not exist. Again, if we have a 1, we x by -1 and add 1 -> 0.
    
            const termTwo = this.labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).add(1e-7).log()); // We did -Actual + 1
    
            const total = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);

            return total;
        })
      

        this.crossEntropyHistory.unshift(cost);



    }

    updateLearningRate() {
        if (this.crossEntropyHistory.length < 2) // enough records to update
            return;

        if (this.crossEntropyHistory[0] > this.crossEntropyHistory[1]) //MSE increased, bad learning rate
            this.options.learningRate /= 2;
        else this.options.learningRate *= 1.05;
    }

    predict(observations) { // observations will HAVE to be in the order of the feature columns - horsepower first.. and so on. -> y= mx+c
        return this.processFeatures(observations).matMul(this.weights).softmax().argMax(1); // Find the maximum value around a row (axis 1) and then returns its index. ArgxMax gives 1 or 0 based on largest arg

    }

}

module.exports = LogisticRegression;