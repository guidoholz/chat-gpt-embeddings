import csv from 'csv-parser';
import * as fs from 'fs';

const results = [];

fs.createReadStream('covid_faq.csv.xls')
  .pipe(csv())
  .on('data', (data) => results.push(data))
  .on('end', () => {
    const res = results.map((item) => {
      return `Q: ${item.questions}\nA: ${item.answers}\n`;
    });
    fs.writeFileSync('context.json', JSON.stringify(res));
  });
