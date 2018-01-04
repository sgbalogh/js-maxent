# js-maxent

A JavaScript (ES6) implementation of a proof-of-concept maximum-entropy (MaxEnt) classifier. When trained on a set of examples, it learns feature weights for classifying inputs with a predicted label (e.g. `"Chicago, Illinois" => "place"`).

This implementation is based on a more complex Java architecture found in an assignment from [Statistical Natural Language Processing Fall 2017](https://cs.nyu.edu/courses/fall17/CSCI-GA.3033-008/) at NYU's Courant Institute.

The library is lightweight enough to use stored weights to classify –– or even run the perceptron to train –– from the DOM.

**TODOs:**
- Create an averaged-perceptron trainer
- Create, or adapt, a [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) gradient descent trainer

Sample invocation:
```javascript
const maxent = require("./maxent.js")
const fs = require("fs")

t = new maxent.Translator()

var training = JSON.parse(fs.readFileSync("./data/train.json")).map( (elem) => { return new maxent.LabeledDatum(elem[1],elem[0])})
var encoded_training = training.map( (elem) => { return new maxent.EncodedDatum(elem,t) })

// Randomize the order of the training data
const shuffleArray = arr => arr.sort(() => Math.random() - 0.5)
encoded_training = shuffleArray(encoded_training)

// Learn weights with the simple perceptron
var cw = maxent.Perceptron.train(t, encoded_training, 1)

// Create a classifier using the weight array and the translator
var classifier = new maxent.Classifier(cw, t)

```

Now you can interact with the classifier like so:
```javascript
console.log(classifier.predict("Chicago, Illinois"))
// This should print 'place'

console.log(classifier.scores("Chicago, Illinois"))
// This should print an array of log probabilities for the labels
```
