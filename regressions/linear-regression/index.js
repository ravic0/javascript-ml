require('@tensorflow/tfjs-node');
// const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
// const Plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
}); // Load csv into code


const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,  // Large Learning rate will cause Overcorrection and ping pong movement -- Try normalization
    iterations: 3,
    batchSize: 10,
});

regression.train();

const r2 = regression.test(testFeatures, testLabels); // coeff of determination -- Only essential during development 

// Plot({
//     x: regression.mseHistory,
//     title: 'my chart',
//     name: 'my_plot'
// });

// console.log(regression.mseHistory);
regression.predict(
    [
        [120,2,380]
    ]
    ).print();

// console.log("Weights: ", regression.weights.print());

console.log("Coefficient of determination: ", r2);

// console.log("M: ", regression.weights.get(1,0), "C: ",regression.weights.get(0,0)); // since padding of 1 is done on the left, weights is [c,m] and not [m,c]