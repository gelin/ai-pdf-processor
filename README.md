# AI PDF Processor

Input: PDF scans (or images) + Questions + Expected Output.

Output: Answers to the questions according to the expected output (boolean, or number, or string).

Uses a small local Llama model to analyse image, recognise text fields and tick boxes to find out the answers.

## API

### Upload file

`PUT|GET|DELETE /file/{name}`

### Ask questions

`POST /file/{name}/ask`, questions, output types: bool, string, number or null.

Result: array of answers of specific types in JSON.

Output type converts to prompt parts.

## Tags

REST API. Python lib to Ollama. Conversion of PDF to PNG.
