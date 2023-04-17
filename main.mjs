import { Configuration, OpenAIApi } from 'openai';
import { get_encoding } from '@dqbd/tiktoken';
import * as fs from 'fs';

const completions_model = 'text-davinci-003';
const embedding_model = 'text-embedding-ada-002';
const embedding_file = 'embeddings.json';
const context_file = 'context.json';

const configuration = new Configuration({
  organization: process.env.OPENAI_ORGANIZATION,
  apiKey: process.env.OPENAI_API_KEY,
});

const queryIndex = process.argv.indexOf('--query');
let queryValue;

if (queryIndex > -1) {
  queryValue = process.argv[queryIndex + 1];
} else {
  console.log('No query provided... use --query "Some questions you have?" to provide a query');
}

const query = queryValue || 'Do I have a question?';

const openai = new OpenAIApi(configuration);

// This function takes an `item` as an input
const get_embedding = async (item) => {
  // Create an embedding for each textblock using the `createEmbedding` method
  const embedding = await openai.createEmbedding({
    input: item,
    model: embedding_model, // The `embedding_model` is the name of the model used to generate the embedding
  });
  // Finally, it returns an object containing the embedding data and the total number of tokens used
  return { embedding: embedding.data.data[0].embedding, tokens: embedding.data.usage.total_tokens };
};

var context = [];
try {
  context = JSON.parse(fs.readFileSync(context_file, 'utf8'));
} catch (error) {
  console.log('Error reading context file', error);
  process.exit();
}

var context_embeddings = [];
try {
  context_embeddings = JSON.parse(fs.readFileSync(embedding_file, 'utf8'));
} catch (error) {
  context_embeddings = await Promise.all(
    context.map(async (item) => {
      return await get_embedding(item);
    })
  );
  fs.writeFileSync(embedding_file, JSON.stringify(context_embeddings));
}

const query_embedding = await get_embedding(query);

// define a function named vector_similarity
const cosine_similarity = (x, y) => {
  // calculate cosine similarity using reduce method
  return Array.from(x).reduce((sum, xi, i) => sum + xi * y[i], 0);
};

// A function that takes Object entries of context embeddings to calculate document similarities
const document_similarities = Object.entries(context_embeddings)
  // For each [doc_index, doc_embedding], calculate cosine_similarity between query and document embeddings
  .map(([doc_index, doc_embedding]) => [cosine_similarity(query_embedding.embedding, doc_embedding.embedding), doc_index])
  // Sorts the similarities in descending order to get the most similar documents at the beginning of the array
  .sort(([similarity1], [similarity2]) => similarity2 - similarity1);

// Maximum allowed length of a section of text
const max_section_len = 1000;

// Separator used to split sections of text
const separator = '\n* ';

// The encoding type used for the GPT-2 text analysis
const encoder = 'gpt2';

// returns the encoding for the specified encoder
const encoding = get_encoding(encoder);

// gets the byte length of the separator string using the specified encoding
const separator_len = Buffer.from(separator, encoding).length;

// An empty array to store chosen sections.
const chosen_sections = [];

// keep track of the token length of the sections in the chosen_sections array.
var chosen_sections_len = 0;

// An empty array to store the indexes of the chosen sections.
const chosen_sections_indexes = [];

// iterate through the array of document similarities and extract the similarity score and document index
for (const [similarity, doc_index] of document_similarities) {
  // get length in bytes of the document content for the current index
  const doc_len = Buffer.from(context[doc_index], encoding).length;

  // check if adding the current document to the list of chosen sections will exceed the maximum section length
  if (chosen_sections_len + doc_len + separator_len < max_section_len) {
    // if adding the current document to the list of chosen sections will not exceed maximum section length,
    // add the document content to the chosen section array, update the chosen sections length, and add the document index to the list of chosen section indexes
    chosen_sections.push(context[doc_index]);
    chosen_sections_len += doc_len + separator_len;
    chosen_sections_indexes.push(doc_index);
  }
}

// Define the header text for the prompt and avoid halucinations
const header = `Answer the question truthfully based on the given context. If you do not find the answer within the text below, respond with 'No idea!'."\n\nContext:\n`;

// Join header, chosen sections and query to create the prompt
const prompt = header + chosen_sections.join('') + '\n\n Q: ' + query + '\n A:';

// Output the entire prompt to the console
console.log('PROMPT: ', prompt);

// create a completion request to query OpenAI's API
const completion = await openai.createCompletion({
  prompt: prompt,
  // set the maximum number of tokens to be generated in the AI's response
  max_tokens: 1000,
  // lower the sampling temperature to reduce the likelihood of the AI generating unexpected responses
  temperature: 0,
  model: completions_model,
});
console.log('\nCOMPLETION ANSWER: ', completion.data.choices[0].text, '\n');
