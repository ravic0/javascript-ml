require('@tensorflow/tfjs-node'); // Use CPU for calculations
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0); //Mean and var of Axis 0

    const scaledPoint = predictionPoint.sub(mean).div(variance.pow(0.5)); // Scaling prediction Point also to the standardized value.

    return features
        .sub(mean)
        .div(variance.pow(0.5)) // Standardize train data
        .sub(scaledPoint) // Pythagoras subtract the differences
        .pow(2) // Pythagoras square the differences (element wise)
        .sum(1) // Add the above in single row -- hence the 1
        .pow(0.5) // Square root -- Shape is 1-D
        .expandDims(1) // Add a column since above chain reduces the shape
        .concat(labels, 1) // Since labels and features are separate and sorting will jumble features and labels
        .unstack() // Explained below since no built in sort methods
        .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1) // swap based on distance
        .slice(0, k) // Get top k values
        .reduce((acc, pair) => acc + pair.get(1), 0) / k;
}

let { features, labels, testFeatures, testLabels } = //features: lat and long dataset except for 10 test features , labels: price , test variables are for test data
    loadCSV('kc_house_data.csv', {
        shuffle: true, // to prevent consecutive data load
        splitTest: 10, // test and training data
        dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living','grade'], // what data to be taken from the csv
        labelColumns: ['price']
    });

features = tf.tensor(features); // Since above values are simple arrays
labels = tf.tensor(labels);
// testFeatures = tf.tensor(testFeatures);
// testLabels = tf.tensor(testLabels);

testFeatures.forEach((testPoint, index) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);

    console.log("Result: ", result, "\nActual Value: ", testLabels[index][0]);

    const err = (testLabels[index][0] - result) / testLabels[index][0];
    console.log("Error: ", err * 100);
    console.log("\n");
})

// console.log(testFeatures);
// console.log(testLabels);


