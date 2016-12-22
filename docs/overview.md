We build this framework with the following design goals in mind.

1. **Ease of use.** One should be able train a  model with single command. The input data are raw text files instead of esoteric file formats. Similarly, using a pre-trained model to make predictions should be straightforward.

2. **Ease to extend**. Code is structured in such a way that it is easy to build upon. For example, adding a new type of attention mechanism or a new encoder architecture requires only minimal code changes.

3. **Well-documented**. In addition to [generated API documentation]() we have written up multiple guides to help users become familiar with the framework.

4. **Good performance**. For the sake of code simplicity we have not tried to squeeze out the last bit of performance, but our implementation is fast enough to cover almost all production use cases. It also supports distributed training to trade off computational power and training time.

5. **Standard Benchmarks**. We provide pre-trained models and benchmark results for several standard datasets. We hope these can  serve as a baseline for further research.
