
const maxent = require("./maxent.js")
const fs = require("fs")

t = new maxent.Translator()

console.log("Loading/encoding training set...")
var training = JSON.parse(fs.readFileSync("./data/train.json")).map( (elem) => { return new maxent.LabeledDatum(elem[1],elem[0])})
var encoded_training = training.map( (elem) => { return new maxent.EncodedDatum(elem,t) })

console.log("Loading/encoding validation set...")
var validation = JSON.parse(fs.readFileSync("./data/validate.json")).map( (elem) => { return new maxent.LabeledDatum(elem[1],elem[0])})
// var encoded_validation = validation.map( (elem) => { return new maxent.EncodedQuery(elem,t) })

console.log("Randomizing order of training data...")
const shuffleArray = arr => arr.sort(() => Math.random() - 0.5)
encoded_training = shuffleArray(encoded_training)

console.log("Learning weights with perceptron...")
var cw = maxent.Perceptron.train(t, encoded_training, 1)

var classifier = new maxent.Classifier(cw, t)

var dev = new maxent.Tester(validation, t, classifier)

dev.accuracy()
