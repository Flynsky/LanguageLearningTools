## How to use Languagedoubler

This is an AI tool that adds translations to e-books in the epub format. Features:

- Translate a full book into a different language
- Translate a full book in a different language and add the original text below each paragraph
- Add a translated paragraph below each original one

The AI uses Ollama,
so it can either use a local AI like translategemma or an API to a cloud AI like GPT or Claude.

![demo picture](./doc/pic/demo.png)

## Get Started 
````
pip install ebooklib beautifulsoup4 requests
curl -fsSL https://ollama.com/install.sh | sh
# Add a translated paragraph below each original one
python doubler.py la_mierda.epub la_mierda_trans.epub --source-lang Spanish --first-lang source --second-lang English --model translategemma:4b
# Translate a full book in a different language and add the original text below each paragraph
python doubler.py Skulduggery_Pleasant_1.epub Skuldugger1_esp.epub --source-lang English --first-lang Spanish --second-lang source --model translategemma:4b
# decent epub reader on linux
foliate out_db.epub
````

## Developer notes


 ## Test Ollama
 ````
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama run qwen2.5:0.5b "Translate to English: Hola, ¿cómo estás?"
````

