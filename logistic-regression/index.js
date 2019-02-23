require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logisticRegression');
const plot = require('node-remote-plot');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: value => value === 'TRUE' ? 1 : 0
    }
});

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.7,
    iterations: 5,
    batchSize: 10,
    decisionBoundary: .5
});

regression.train();

console.log(
    regression.test(testFeatures, testLabels)
);

plot({
    x: regression.costHistory.reverse()
})

regression.predict([
    [130, 307, 1.75],
    [113, 95, 1.19]
]).print();
