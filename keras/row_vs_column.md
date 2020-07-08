# RNN on MNIST: row-first v.s. column-first

## Background
It runs RNN on the MNIST 28x28 images, so there are two options of row-first
and column first.

## Observation

* The column-first optimizes faster, because it always outperforms at the first
epoch.

* The column-first also outperforms a little bit in the end.

* The gap is larger on SimpleRNN than LSTM.
