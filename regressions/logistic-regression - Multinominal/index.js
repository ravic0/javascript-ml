require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
// const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');
const mnist = require('mnist-data');


function loadData(){
    const mnistData = mnist.training(0, 60000); // release reference to save memory for Garbage collector

    const features = mnistData.images.values.map(img => _.flatMap(img));
    
    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0); // fill row with 0
        row[label] = 1;
        return row;
    }); // creating our row of labels

    return {features, encodedLabels}
}

const {features, encodedLabels} = loadData();

const regression = new LogisticRegression(features, encodedLabels, {
    learningRate: 1,
    iterations: 50,
    batchSize:500
});

regression.train();

const testMnistData = mnist.testing(0,10000);

const testFeatures = testMnistData.images.values.map(img => _.flatMap(img));

const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0); // fill row with 0
    row[label] = 1;
    return row;
}); // creating our row of labels

const accuracy = regression.test(testFeatures, testEncodedLabels);

console.log("Accuracy: ",accuracy);








// const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
//     dataColumns: ['horsepower','displacement', 'weight'],
//     labelColumns: ['mpg'],
//     shuffle: true,
//     splitTest: 50,
//     converters: {
//         mpg: (value) => {
//             const mpg = parseFloat(value); // ENCODING -- order will be lost since shuffle is true
//             if(mpg < 15)
//             return [1,0,0]; // low, medium, high
//             else if(mpg < 30)
//             return [0,1,0];
//             else return [0, 0, 1];
//         }
//     }
// });

 
// const regression = new LogisticRegression(features, _.flatMap(labels), { // multiple labels for classification
//     learningRate: 0.5,
//     iterations: 100,
//     batchSize: 50,
//     decisionBoundary: 0.4
// });

// regression.train();

// console.log(regression.test(testFeatures, _.flatMap(testLabels)));

// regression.predict([[
//     150, 200, 2.332
// ]]).print();
