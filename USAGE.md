If you want to use my example please follwo the steps:

### Export your API Key and Organization Key

```bash
export OPENAI_API_KEY=sk-XXXXXXXXXXXXX....
export OPENAI_ORGANIZATION='org-XXXXXXXX....
```

If you don't have the access right now, create an account on OpenAIs Webpage to get the keys.

### Run the example

run:

```bash
npm install
```

```bash
node main.mjs --query "What is COVID 19?"
```

or

```bash
node main.mjs --query "Where is the Eifeltower?"
```

If you want to change to a context you are interested in, just delete the _embeddings.json_ and create a _context.json_, where a list of textblocks are given.
Run the example, and the _embeddings.json_ are automatically created.
