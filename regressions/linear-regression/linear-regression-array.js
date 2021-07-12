const tf = require('@tensorflow/tfjs'); // No tfjs-node again
const _ = require('lodash');

const defaultOptions = { learningRate: 0.1, iterations: 1000 }; //'iterations' to disallow training to keep running forever.

class LinearRegression {

    constructor(features, labels, options) {
        this.features = features;
        this.labels = labels;
        this.options = Object.assign({ ...defaultOptions }, options);

        this.m = 0; // Initial guesses for m and c
        this.c = 0;
    }

    train() {
        for(let i=0; i<this.options.iterations ; i++)
        {
            this.gradientDescent();
        }
    }

   // -- Array only implementation (supports only 1 feature i.e x )

    gradientDescent() {                               
        // Calculate MSE and its slope wrt c and m

        const currentGuessesForMPG = this.features.map(row => // Calculate mx(i) + c -- row is the feature and is iterated for the entire length of features.
            this.m * row[0] + this.c
         );

        const cSlope = _.sum(currentGuessesForMPG.map((guess, index) =>
            guess - this.labels[index][0]
        )) * 2 / this.features.length;

        const mSlope = _.sum(currentGuessesForMPG.map((guess, index) => 
         -1 * this.features[index][0] * (this.labels[index] - guess)
        )) * 2 / this.features.length;

        this.m = this.m - mSlope * this.options.learningRate;

        this.c = this.c - cSlope * this.options.learningRate;
        
    }
}

module.exports = LinearRegression;