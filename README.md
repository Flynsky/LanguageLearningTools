## How to use Languagedoubler
````
pip install
sudo dnf install ollama
ollama pull 
python doubler.py ./example/Imperio\ Final.\ \(Ed.\ revisada\),\ El\ -\ Brandon\ Sanderson.epub out_db.epub --model qwen2.5:0.5b --source-lang Espan
iol  --title "DoubleLanguage" --debug --translation-italic --translation-small
foliate out_db.epub
````

 ## Test Ollama
 ````
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull qwen2.5:0.5b
ollama run qwen2.5:0.5b "Translate to English: Hola, ¿cómo estás?"
````

