## How to use Languagedoubler
````
pip install
sudo dnf install ollama
ollama pull 
# adding translatinos to a spanish book 
python doubler.py la_mierda.epub la_mierda_trans.epub --source-lang Spanish --first-lang source --second-lang English --model translategemma:4b
# translate an englisch book to Spanish<->English
python doubler.py Skulduggery_Pleasant_1.epub Skuldugger1_esp.epub --source-lang English --first-lang Spanish --second-lang source --model translategemma:4b
foliate out_db.epub
````

 ## Test Ollama
 ````
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama run qwen2.5:0.5b "Translate to English: Hola, ¿cómo estás?"
````

