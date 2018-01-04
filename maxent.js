
var FeatureExtractor = class FeatureExtractor {
  // This is a sample (and not very good) feature-
  // extraction method; replace this with any other
  // method that returns an object of feature-count
  // pairs
  static extract(labeled_entry) {
    const datum = labeled_entry.datum
    var features = {}

    if (datum.indexOf("-") > -1) {
      features["contains-dash"] = 1
    }

    var matches = datum.match(/\d+/g);
    if (matches != null) {
      features["contains-number"] = 1
    }

    matches = datum.match(/\.+/g);
    if (matches != null) {
      features["contains-period"] = 1
    }

    matches = datum.match(/,+/g);
    if (matches != null) {
      features["contains-comma"] = 1
    }

    var tokens = datum.split(" ")

    features["NUM-TOKENS"] = tokens.length

    var title = 0
    var not_title = 0

    tokens.forEach( (t) => {
      if (/^[A-Z]/.test( t) ) {
        title += 1
      } else {
        not_title += 1
      }

      var k2 = "SUF-2-" + t.slice(-2)
      FeatureExtractor.update(features, k2, 1)

      k2 = "SUF-1-" + t.slice(-1)
      FeatureExtractor.update(features, k2, 1)

      k2 = "SUF-3-" + t.slice(-3)
      FeatureExtractor.update(features, k2, 1)

      k2 = "SUF-4-" + t.slice(-4)
      FeatureExtractor.update(features, k2, 1)

      k2 = "PRE-1-" + t.slice(0,1)
      FeatureExtractor.update(features, k2, 1)

      k2 = "PRE-2-" + t.slice(0,2)
      FeatureExtractor.update(features, k2, 1)

      k2 = "PRE-3-" + t.slice(0,3)
      FeatureExtractor.update(features, k2, 1)

      k2 = "PRE-4-" + t.slice(0,4)
      FeatureExtractor.update(features, k2, 1)
    })

    if (title > 0 && not_title == 0) {
      features["ALL-TITLE"] = 1
    }

    if (title > 0 && not_title > 0) {
      features["SOME-TITLE-SOME-NOT"] = 1
    }

    if (title == 0 && not_title > 0) {
      features["ALL-NOT-TITLE"] = 1
    }

    FeatureExtractor.gram(datum, 3).forEach( (x) => {
      var k2 = "TRI_GRAM_" + x
      FeatureExtractor.update(features, k2, 1)
    } )

    FeatureExtractor.gram(datum, 2).forEach( (x) => {
      var k2 = "BI_GRAM_" + x
      FeatureExtractor.update(features, k2, 1)
    } )

    return features
  }

  // Takes a collection of features (object), and
  // a key/value for update
  static update(collection, k_name, v) {
    if (k_name in collection) {
      collection[k_name] += v
    } else {
      collection[k_name] = v
    }
  }

  // Extract n-grams of arbitrary size, and return
  // them as a list
  static gram(input, n) {
    var grams = []
    for (var a = 0; a < input.length; a++) {
      grams.push(input.slice(a,a+n))
    }
    return grams
  }

}

// A wrapper object for a labeled datum with
// extracted features
var EncodedDatum = class EncodedDatum {
  constructor(labeled_datum, translator) {
    this.feature_indices = []
    this.feature_counts = []
    this.true_class = null
    Object.entries(FeatureExtractor.extract(labeled_datum)).forEach(
      ([key, value]) => {
      const k_idx = translator.getFeatureIndex(key)
      this.feature_indices.push(k_idx)
      this.feature_counts.push(value)
    })
    if (labeled_datum.label != null) {
      this.true_class = translator.getLabelIndex(labeled_datum.label)
    }
  }
  getNumFeaturesPresent() {
    return this.feature_indices.length
  }
  getFeatureIndex(arr_idx) {
    return this.feature_indices[arr_idx]
  }
  getFeatureCount(arr_idx) {
    return this.feature_counts[arr_idx]
  }
}

// Similar to EncodedDatum, but for inputs that do not
// have a known label (i.e., test data)
var EncodedQuery = class EncodedQuery {
  // If new features are found during the extraction pro-
  // cess, they will not affect the translator
  constructor(labeled_datum, translator) {
    this.feature_indices = []
    this.feature_counts = []
    Object.entries(FeatureExtractor.extract(labeled_datum)).forEach(
      ([key, value]) => {
      const k_idx = translator.getFeatureIndex(key, false)
      if (k_idx > -1) {
        this.feature_indices.push(k_idx)
        this.feature_counts.push(value)
      }
    })
    if (labeled_datum.label != null) {
      this.true_class = translator.getLabelIndex(labeled_datum.label, false)
    }
  }
  getNumFeaturesPresent() {
    return this.feature_indices.length
  }
  getFeatureIndex(arr_idx) {
    return this.feature_indices[arr_idx]
  }
  getFeatureCount(arr_idx) {
    return this.feature_counts[arr_idx]
  }
}

// Contains a single static method for determining
// log probabilities of labels for an encoded datum
// or encoded query
var LogProb = class LogProb {
  static getLogProbs(encoded_datum, weights, translator) {
    const activations = []
    const log_probabilities = []

    // Initialize activations
    for (var x = 0; x < translator.getNumLabels(); x++) {
      activations.push(0)
      log_probabilities.push(Number.NEGATIVE_INFINITY)
    }

    var all_activations = 0
    for (var i = 0; i < translator.getNumLabels(); i++) {
      for (var j = 0; j < encoded_datum.getNumFeaturesPresent(); j++) {
        var feature_idx = encoded_datum.getFeatureIndex(j)
        var feature_count = encoded_datum.getFeatureCount(j);
        var linear_index = feature_idx + (i * translator.getNumFeatures())
        if (linear_index < weights.length) {
          var relevant_weight = weights[linear_index]
        } else {
          var relevant_weight = 0
        }
        activations[i] += relevant_weight * feature_count
      }
      all_activations += activations[i]
    }
    for (var i = 0; i < translator.getNumLabels(); i++) {
      if (all_activations > 0) {
        log_probabilities[i] = Math.log(activations[i] / all_activations);
      }
    }
    return log_probabilities
  }
}

// A simple non-averaging perceptron trainer
var Perceptron = class Perceptrion {
  static train(translator, encoded_datum_list, iters=1) {
    const shuffleArray = arr => arr.sort(() => Math.random() - 0.5)

    // Initialize weights
    var weights = []
    for (var i = 0; i < translator.getNumLinearIndex(); i++) {
      weights.push(1.0);
    }

    // Begin iterating through training corpus
    for (var iter = 0; iter < iters; iter++) {
      encoded_datum_list = shuffleArray(encoded_datum_list)
      var correct_guesses = 0
      var incorrect_guesses = 0

      // Begin iterating through data
      for (var d = 0; d < encoded_datum_list.length; d++) {
        var datum = encoded_datum_list[d];
        var log_probs = LogProb.getLogProbs(datum, weights, translator);
        var true_class = datum.true_class;
        var predicted_winner = log_probs.indexOf(Math.max(...log_probs))

        if (true_class != predicted_winner) {
          incorrect_guesses += 1
          for (var c = 0; c < datum.getNumFeaturesPresent(); c++) {
            var correct_lin_idx = (true_class * translator.getNumFeatures()) + datum.getFeatureIndex(c);
            var incorrect_lin_idx = (predicted_winner * translator.getNumFeatures()) + datum.getFeatureIndex(c);
            var feature_count = datum.getFeatureCount(c);
            weights[correct_lin_idx] += feature_count
            weights[incorrect_lin_idx] -= feature_count
          }
        } else {
          correct_guesses += 1
        }
      }
      console.log("Finished iter: " + iter)
      console.log("Training accuracy: " + (correct_guesses / (correct_guesses + incorrect_guesses)))
    }
    return weights
  }

}

// A wrapper class for making label predictions
var Classifier = class Classifier{
  constructor(weights, translator) {
    this.weights = weights
    this.translator = translator
  }
  predict(input) {
    var le = new LabeledDatum(input, null)
    var ee = new EncodedQuery(le, this.translator)
    var lp = LogProb.getLogProbs(ee, this.weights, this.translator)
    var p_class = lp.indexOf(Math.max(...lp))
    return this.translator.getLabel(p_class)
  }
  scores(input) {
    var le = new LabeledDatum(input, null)
    var ee = new EncodedQuery(le, this.translator)
    var lp = LogProb.getLogProbs(ee, this.weights, this.translator)
    return lp
  }
}

// Maps features and labels to indices and vice-versa
var Translator = class Translator {
  constructor() {
    this.features = []
    this.labels = []
  }
  getNumLinearIndex() {
    return this.getNumFeatures() * this.getNumLabels();
  }

  getNumFeatures() {
    return this.features.length
  }

  getNumLabels() {
    return this.labels.length
  }

  getFeatureIndex(feature_name, write=true) {
    var resp = this.features.indexOf(feature_name)
    if (resp > -1) {
      return resp
    } else {
      if (write) {
        this.features.push(feature_name)
        return this.features.length - 1
      } else {
        return -1
      }
    }
  }

  getLabelIndex(label_name, write=true) {
    var resp = this.labels.indexOf(label_name)
    if (resp > -1) {
      return resp
    } else {
      if (write) {
        this.labels.push(label_name);
        return this.labels.length - 1;
      } else {
        return -1;
      }
    }
  }

  getLabel(label_index) {
    if (label_index > -1 && label_index < this.labels.length) {
      return this.labels[label_index]
    } else {
      return null
    }
  }

  getFeature(feature_index) {
    if (feature_index > -1 && feature_index < this.features.length) {
      return this.features[feature_index]
    } else {
      return null
    }
  }

}

var LabeledDatum = class LabeledDatum {
  constructor(datum, label=null) {
    this.label = label;
    this.datum = datum;
  }
}

var Tester = class Tester {
  constructor(labeled_data, translator, classifier) {
    this.translator = translator
    this.classifier = classifier
    this.labeled = labeled_data
  }

  accuracy() {
    var correct = 0
    var incorrect = 0
    this.labeled.forEach( (elem) => {
      if (elem.label == this.classifier.predict(elem.datum)) {
        correct += 1
      } else {
        incorrect += 1
      }
    })
    console.log("Correct: " + correct + "\nIncorrect: " + incorrect + "\n\nAccuracy: " + correct / (correct + incorrect))
  }
}

module.exports = {
  LabeledDatum: LabeledDatum,
  EncodedDatum: EncodedDatum,
  FeatureExtractor: FeatureExtractor,
  Translator: Translator,
  Perceptron: Perceptron,
  Classifier: Classifier,
  LogProb: LogProb,
  EncodedQuery: EncodedQuery,
  Tester: Tester
}
