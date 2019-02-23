require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linearRegression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1, // 0.1 optimal
    iterations: 3, // 7 optimal
    batchSize: 10 // 9 optimal
});


regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'MSE'
})
console.log(' R2 is: ', r2);

regression.predict([
    [130, 1.75, 307],
    [165, 1.85, 350],
    [150, 1.72, 318]
]).print();
