# Power of Embeddings

## TL;DR

- We discuss the question **_what are embeddings_** and **_how do I use embeddings_**?
- Embeddings are used to create contexts for a questions/tasks on language models such as GPT-3/4 without having to train the language model beforehand through fine-tuning.
- Embeddings are not only useful for language models.
- In the following, we use the OpenAI API with Javascript to create the embeddings and to answer our context-related questions.

### ---

Many have already tested the power of OpenAI's new language models. Many have already tried the first steps of the API connection of OpenAI. In addition to a normal connection of the API to the completion, there is the possibility to feed GPT with information so that the model can provide an answer tailored to you. OpenAI offers two possibilities here:

1.  ### Finetuning

    As the name suggests, you have the possibility to train an existing model by finetuning your own model. For more information on finetuning, have a look at [OpenAI](https://platform.openai.com/docs/guides/fine-tuning).

2.  ### Embeddings
    Embeddings are the representations of words in an n-dimensional vector space, whereby semantic information of the text is taken into account.
    Texts (from a sentence to a paragraph) can thus be compared with each other as to whether and in which semantic context they are related. The **_similarity_** of the vectors to each other determines the semantic affinity. You can find a good explanation [here](https://saschametzger.com/blog/a-beginners-guide-to-tokens-vectors-and-embeddings-in-nlp/).

In this project I show how to use embeddings with and through OpenAI. At the end, you will be able to create embeddings yourself and use them to generate a model that is tailored to your needs without any fine-tuning. For this I use the [FAQ data set for COVID 19 by Kaggle](https://www.kaggle.com/datasets/deepann/covid19-related-faqs)

I am dividing the project into 4 sections:

1. ### Creating the embeddings

   We want to map the following text as an embedding:

   ```
   Q: What is a novel coronavirus?
   A: A novel coronavirus is a new coronavirus that has not been previously identified. The virus causing coronavirus disease 2019 (COVID-19), is not the same as the coronaviruses that commonly circulate among humans and cause mild illness, like the common cold.
   ```

   A multi-dimensional vector is now calculated for the entire text block

   ```javascript
   const get_embedding = async (item) => {
     const embedding = await openai.createEmbedding({
       input: item,
       model: 'text-embedding-ada-002',
     });
     return { embedding: embedding.data.data[0].embedding, tokens: embedding.data.usage.total_tokens };
   };
   ```

   As a result, we get a multidimensional vector and the number of underlying tokens (word groups).

   ```json
   {
     "embedding": [-0.0009364469,-0.01185554,0.007705795,0.0015959381,-0.0033754932,-0.008391298,-0.025363633,-0.004039575,-0.03988162,0.026759123,0.012479838,0.005765575,-0.03349175,0.02391918,-0.006108327,0.0032745039,0.026979463,-0.025608456,0.034446556,-0.03136179,-0.015166767,0.014383335,-0.02491071,...],"tokens": 63
   }
   ```

   Now we know the number of tokens and the position of the vector. If we have two embeddings, we can compare them by cosine similarity. But more about that later.
   We use the API of OpenAI to calculate the embeddings. But you can also use your own local model or one of the well-known services like [pinecone.io](https://www.pinecone.io).

   We now have a set of contexts (question and answer in our example) and their corresponding vectors.

2. ### Calculate the _similarity_

   Without going too deep into the matter, we now want to find one or more embeddings (in our case FAQ sections) that could answer our question.

   Let's say our question is:

   ```
   Was genau ist COVID 19?
   ```

   First, we calculate our multi-dimensional vector for the question. We now want to compare this with the embeddings/vectors already calculated above. For this we use the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

   ```javascript
   const cosine_similarity = (question, embedding_X) => {
     // calculate cosine similarity using reduce method
     return Array.from(question).reduce((sum, xi, i) => sum + xi * embedding_X[i], 0);
   };
   ```

   As a result we get a value between 0 and 1 (more precisely, abs(-1 and 1) ). The closer to 0 the more the context of the question matches the context of the embedding.

   **In fact, with a few lines of code, we have laid the foundation of a chat bot for e.g. FAQs :)**.

   If we now sort the values in ascending order, the result will be "**all context entries sorted by relevance to the question**".

3. ### Selection of embeddings

   We are almost there. Now we select a set of embeddings that we send to the API to answer our question. Since we want to be economical, but also the OPenAI API does not let us send an infinite amount of context, we want to send embeddings that do not exceed a certain amount of tokens (we remember above, we saved the tokens above).

   Let's say we don't want our request to exceed 1000 tokens, we iterate over our list of sorted embeddings and include every embedding as long as it doesn't exceed the maximum amount of tokens.

   ```javascript
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
   const encoding = get_encoding(encoder); // npm package by '@dqbd/tiktoken'

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
   ```

   As a result, we get a few embeddings, which we now send to the API in our last step to answer our question.

4. ### Question to the API

   Before we compose our context for the question, we would like to point out to GPT not to answer questions that it cannot answer or that do not belong to our context. We can prevent the so-called **_hallucination_** by adding the following sentence to our context:

   ```
   Answer the question truthfully based on the given context. If you do not find the answer within the text below, respond with 'No idea!'
   ```

   If we now compose our prefix, embeddings and question, we get:

   ```
   PROMPT:  Please answer the question truthfully based on the given context. If you do not find the answer within the text below, respond with 'No idea!'."

   Context:
   Q: How can I prepare for COVID-19 at work?
   A: Plan for potential changes at your workplace. Talk to your employer about their emergency operations plan, including sick-leave policies and telework options. Learn how businesses and employers can plan for and respond to COVID-19.
   Q: Is it possible to have the flu and COVID-19 at the same time?
   A: Yes. It is possible to test positive for flu (as well as other respiratory infections) and COVID-19 at the same time.
   Q: Who is at higher risk for serious illness from COVID-19?
   A: COVID-19 is a new disease and there is limited information regarding risk factors for severe disease. Based on currently available information and clinical expertise, older adults and people with underlying medical conditions are at higher risk for severe illness from COVID-19.
   Q: How can I protect myself?
   A: Visit the How to Protect Yourself & Others page to learn about how to protect yourself from respiratory illnesses, like COVID-19.


   Q: what is covid 19?
   A:
   ```

   This text block will now be sent to the API

   ```javascript
   const completion = await openai.createCompletion({
     prompt: prompt,
     // set the maximum number of tokens to be generated in the AI's response
     max_tokens: 1000,
     // lower the sampling temperature to reduce the likelihood of the AI generating unexpected responses
     temperature: 0,
     model: 'text-davinci-003',
   });
   console.log('\nCOMPLETION ANSWER: ', completion.data.choices[0].text, '\n');
   ```

   As a result we get:

   ```
   COMPLETION ANSWER:   COVID-19 is a new disease caused by a novel (or new) coronavirus that has not previously been seen in humans. The virus is thought to spread mainly from person-to-person contact, including through respiratory droplets produced when an infected person coughs or sneezes.
   ```

   Let us now ask a question that is not in our context

   ```
   Question: Where is the Eifeltower?
   ```

   we receive in response:

   ```
   COMPLETION ANSWER:   No idea!
   ```

**_Voilà_**

We have seen how we can use embeddings, how they are generated and how their similarities to each other are calculated. We have seen how we can use the OpenAI API to do this and how we can answer a context for a question in context using the Open AI API.
