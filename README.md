# Huggingface Transformers in NodeJS

This library is a fork of [praeclarum/transformers-js](https://github.com/praeclarum/transformers-js), it's compatible with NodeJS and it's up to date. 

# Usage

This isn't hard.

```js
import transformers from 'transformers-nodejs';

const tokenizer = await transformers.AutoTokenizer.fromPretrained('./tokenizer.json');
const model = await transformers.AutoModelForSeq2SeqLM.fromPretrained('./model.onnx');

const inputTokenIds = tokenizer.encode('Hello!');
const outputTokenIds = await model.generate(inputTokenIds, {maxLength:54,topK:10});
const output = tokenizer.decode(outputTokenIds, true);
```
