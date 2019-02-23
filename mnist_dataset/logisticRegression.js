const tf = require('@tensorflow/tfjs');

const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options) { // assuming features ad labels are tensors
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);

        this.options = Object.assign({ learningRate: 0.1, decisionBoundary: 0.5, iterations: 1000 }, options);

        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]); // m and b
        this.costHistory = []; // cross entropy history
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).softmax(); // it using cross entropy becaues mse is good for logistic regression
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);

        return this.weights.sub(slopes.mul(this.options.learningRate));
    }

    train() {
        const batchQuantity = Math.floor(
            this.features.shape[0] / this.options.batchSize
        );

        for (let i = 0; i < this.options.iterations; i++) {

            for (let j = 0; j < batchQuantity; j++) {

                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;
                this.weights = tf.tidy(() => {
                    const featureSlice = this.features
                        .slice([startIndex, 0], [batchSize, -1]);
                    const labelSlice = this.labels
                        .slice([startIndex, 0], [batchSize, -1]);

                    return this.gradientDescent(featureSlice, labelSlice);
                });
            }
            this.recordCost();
            this.updateLearningRate();
        }
    }

    predict(observations) {
        return this.processFeatures(observations)
            .matMul(this.weights)
            .softmax()
            .argMax(1);
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);

        testLabels = tf.tensor(testLabels).argMax(1);

        const incorrect = predictions
            .notEqual(testLabels)
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
        const filler = variance.cast('bool').logicalNot().cast('float32');
        this.mean = mean;

        this.variance = variance.add(filler); // convert all zeros inside variance into '1'

        return this.standardization(features);
    }
    standardization(features) {
        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    recordCost() {
        const cost = tf.tidy(() => {
            
            const guesses = this.features.matMul(this.weights).softmax();

            const termOne = this.labels
                .transpose()
                .matMul(guesses
                    .add(1e-7) // make sure we never take log of zero, instead we take log of a value very close to zero
                    .log());
            const termTwo = this.labels
                .mul(-1)
                .add(1)
                .transpose()
                .matMul(
                    guesses.mul(-1)
                    .add(1)
                    .add(1e-7) // make sure we never take log of zero, instead we take log of a value very close to zero
                    .log()
                );

            return termOne.add(termTwo)
                .div(this.features.shape[0])
                .mul(-1)
                .get(0, 0);
        })
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