const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options) { // assuming features ad labels are tensors
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);

        this.options = Object.assign({ learningRate: 0.1, decisionBoundary: 0.5, iterations: 1000 }, options);

        this.weights = tf.zeros([this.features.shape[1], 1]); // m and b
        this.costHistory = []; // cross entropy history
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).sigmoid();
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    train() {
        const batchQuantity = Math.floor(
            this.features.shape[0] / this.options.batchSize
        );

        for (let i = 0; i < this.options.iterations; i++) {

            for (let j = 0; j < batchQuantity; j++) {
                
                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;
                const featureSlice = this.features
                    .slice([startIndex, 0], [batchSize, -1]);
                const labelSlice = this.labels
                    .slice([startIndex, 0], [batchSize, -1]);
                
                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordCost();
            this.updateLearningRate();
        }
    }

    predict(observations) {
        return this.processFeatures(observations)
            .matMul(this.weights).sigmoid()
            .greater(this.options.decisionBoundary)
            .cast('float32');

    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
            
        testLabels = tf.tensor(testLabels);

        const incorrect = predictions.sub(testLabels)
            .abs()
            .sum()
            .get();
        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    processFeatures(features) {
        features = tf.tensor(features);

        if (this.mean && this.variance) {
            features = this.standardization(features);
        } else {
            features = this.standardize(features);
        }

        features = tf.ones([features.shape[0], 1]).concat(features, 1); // appending column of 1s in features

        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return this.standardization(features);
    }
    standardization(features) {
        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    recordCost() {
        const guesses = this.features.matMul(this.weights).sigmoid();

        const termOne = this.labels
            .transpose()
            .matMul(guesses.log());
        const termTwo = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(
                guesses.mul(-1).add(1).log()
            );
        
        const cost = termOne.add(termTwo)
                .div(this.features.shape[0])
                .mul(-1)
                .get(0, 0);
        this.costHistory.unshift(cost); // cross entropy
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) {
            return;
        }

        if (this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }
    // gradientDescent() {
    //     const currentGuessesForMPG = this.features.map(row => this.m * row[0] + this.b);// m * x + b
    //     const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return guess - this.labels[i][0];
    //     })) * 2 / this.features.length;

    //     const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return -1 * this.features[i][0] * (this.labels[i][0] - guess);
    //     })) * 2 / this.features.length;

    //     this.m = this.m - mSlope * this.options.learningRate;
    //     this.b = this.b - bSlope * this.options.learningRate;
    // }
}


module.exports = LogisticRegression;