import ort from 'onnxruntime-node';

class PretrainedModel {
	static async loadSession(modelSource) {
		const session = await ort.InferenceSession.create(modelSource);
		return session;
	}
}

class AutoModelForSeq2SeqLM extends PretrainedModel {
	constructor(encoderSession, initDecoderSession, decoderSession) {
		super();
		this.encoderSession = encoderSession;
		this.initDecoderSession = initDecoderSession;
		this.decoderSession = decoderSession;
	}

	static async fromPretrained(model) {
		const encoderSession = await this.loadSession(model);

		return new T5ForConditionalGeneration(encoderSession);
	}

	/**
   * Generate a sequence of tokens.
   *
   * @param {Array} inputTokenIds
   * @param {Object} options Properties:
   *  `maxLength` for the maximum generated sequence length,
   *  `topK` for the number of logits to consider when sampling.
   * @param {Promise} progressAsyncCallback
   * @returns The generated sequence of tokens.
   */
	async generate(inputTokenIds, options, progressAsyncCallback) {
		const maxLength = options.maxLength || 100;
		const topK = options.topK || 0;
		const startOfDecoderTokenId = 0;
		const endOfDecoderTokenId = 1;
		let outputTokenIds = [startOfDecoderTokenId];
		let numOutputTokens = 1;
		let shouldContinue = true;
		const maxOutputTokens = numOutputTokens + maxLength;
		async function progress() {
			if (progressAsyncCallback) {
				shouldContinue = await progressAsyncCallback(
					outputTokenIds,
					inputTokenIds
				);
			}
		}
		let sampler = (x) => this.sampleLogitsGreedily(x);
		if (topK > 0) {
			sampler = (x) => this.sampleLogitsTopK(x, topK);
		}
		while (shouldContinue && numOutputTokens < maxOutputTokens) {
			let output = await this.forward(inputTokenIds, outputTokenIds);
			let newTokenId = sampler(output.logits);
			outputTokenIds.push(newTokenId);
			numOutputTokens++;
			await progress(outputTokenIds);
			if (newTokenId === endOfDecoderTokenId) {
				break;
			}
		}
		return outputTokenIds;
	}

	sampleLogitsGreedily(logits) {
		let shape = logits.dims;
		let [batchSize, seqLength, vocabSize] = shape;
		let n = batchSize * seqLength * vocabSize;
		let startIndex = n - vocabSize;
		let argmaxi = 0;
		let argmax = logits.data[startIndex + argmaxi];
		for (let i = 1; i < vocabSize; i++) {
			let l = logits.data[startIndex + i];
			if (l > argmax) {
				argmaxi = i;
				argmax = l;
			}
		}
		return argmaxi;
	}
	sampleLogitsTopK(logits, k) {
		let shape = logits.dims;
		let [batchSize, seqLength, vocabSize] = shape;
		let n = batchSize * seqLength * vocabSize;
		let startIndex = n - vocabSize;
		let logs = logits.data.slice(startIndex);
		k = Math.min(k, vocabSize);
		let logitAndId = Array.from(logs)
			.map((x, i) => [x, i])
			.sort((a, b) => b[0] - a[0]);
		const sMin = Math.exp(-100.0);
		let sumS = 0.0;
		for (let i = 0; i < logitAndId.length; i++) {
			const s = i < k ? Math.exp(logitAndId[i][0]) : sMin;
			sumS += s;
			logitAndId[i][0] = s;
		}
		let r = Math.random() * sumS;
		for (let i = 0; i < logitAndId.length; i++) {
			r -= logitAndId[i][0];
			if (r <= 0) {
				return logitAndId[i][1];
			}
		}
		return logitAndId[0][1];
	}
}

class T5ForConditionalGeneration extends AutoModelForSeq2SeqLM {
	constructor(encoderSession) {
		super(encoderSession);
	}

	async forward(inputIds, decoderInputIds) {
		const inputIdsTensor = new ort.Tensor(
			'int64',
			new BigInt64Array(inputIds.map((x) => BigInt(x))),
			[1, inputIds.length]
		);
		const encoderAttentionMaskTensor = new ort.Tensor(
			'int64',
			new BigInt64Array(inputIds.length).fill(1n),
			[1, inputIds.length]
		);
		const decoderInputIdsTensor = new ort.Tensor(
			'int64',
			new BigInt64Array(decoderInputIds.map((x) => BigInt(x))),
			[1, decoderInputIds.length]
		);
		const decoderAttentionMaskTensor = new ort.Tensor(
			'int64',
			new BigInt64Array(decoderInputIds.length).fill(1n),
			[1, decoderInputIds.length]
		);

		const encoderFeeds = {
			input_ids: inputIdsTensor,
			attention_mask: encoderAttentionMaskTensor,
			decoder_input_ids: decoderInputIdsTensor,
			decoder_attention_mask: decoderAttentionMaskTensor
		};

		const encoderResults = await this.encoderSession.run(encoderFeeds);
		const logits = encoderResults.logits;
		const encoderHiddenStates = encoderResults.hidden_states;
		const encoderOutputs = encoderHiddenStates;
		return new Seq2SeqLMOutput(logits, encoderOutputs);
	}
}

class Seq2SeqLMOutput {
	constructor(logits, encoderOutputs) {
		this.logits = logits;
		this.encoderOutputs = encoderOutputs;
	}
}

export default AutoModelForSeq2SeqLM;
