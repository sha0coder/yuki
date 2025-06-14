'''
    coder agentic AI 100% offline
    voice instructions + system prompt -> stt -> AI -> json -> parser -> python function or cmd
'''


from TTS.api import TTS
import whisper
import sounddevice as sd
import numpy as np
import ollama
import logging
import sys
import re
import os
import os.path
import warnings
import json
import subprocess
from contextlib import redirect_stdout, redirect_stderr
from pygments import highlight
from pygments.lexers import get_lexer_for_filename
from pygments.formatters import TerminalFormatter

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if len(sys.argv) > 1 and 'go' in sys.argv:
    GO_AHEAD = True
else:
    GO_AHEAD = False


VOICE = True
SAMPLERATE = 16000
VOICE  = 'tts_models/en/ljspeech/vits' #'tts_models/en/vctk/vits'
MODEL = 'qwen2.5-coder:latest' #'mbenhamd/qwen2.5-7b-instruct-cline-128k-q8_0:latest' #'wizardlm2'  # 'qwen2.5-coder:latest'
WHISPER_MODEL = 'turbo' #'large-v3-turbo' #tiny" #"turbo"
MODELS = [
    'qwen2.5-coder:latest',
    'mbenhamd/qwen2.5-7b-instruct-cline-128k-q8_0:latest',
    'llama2-uncensored',
    'wizardlm2',
]
ENERGY_THRESHOLD = 0.02  # Umbral de energía para detectar voz
SILENCE_DURATION = 4  # Segundos de silencio antes de detener la grabación
CHUNK_DURATION = 0.1  # Duración de cada fragmento de audio (100 ms)
MAX_DURATION = 40  # Duración máxima de grabación (segundos)

model = whisper.load_model(WHISPER_MODEL, device="cpu")
context = ''

if len(sys.argv) > 1 and 'ctx' in sys.argv:
    context = open('ctx').read()

SYSTEM_PROMPT = '''
You are Yuki, a linux terminal, and software engineer.
You are going to use function calling to execute yuki commands in order to create the project structure and also to create the code that satisfy the User Request.
Note that the user input come from speech to text and some words could not be exactly what it should be.

Discover the files of the project, if there are files view the contect, and if are empty, fill its content, if dont exist create them.

IMPORTANT: Dont return multiple jsons, do things step by step, return only the next step, dont overthink.

## Function Calling

Respond ONLY using 1 valid JSON without using markdown. You are designed to process user queries and decide if a local function needs to be executed. Follow these steps:

1. Analyze the user input to determine if it requires invoking a local function or just returning a direct response.
2. If it requires a function call:
 - Use the key "action": "yuki".
 - Provide the "command" as a string.
 - Optionally, provide a "description" to summarize your intent.
3. If we think we know the answer and no function call is required:
 - Use the key "action": "reply".
 - Include "response" with the direct answer to the user query.
4. If the program is done and run well without errors:
 - use the key "action": "done".
 - add a "description"
5. if you need to edit a program line, this is not to add code, is jut to edit existing line.
 - use "action": "edit"
 - add code in "code": "put code here"
 - indicate the line in "line": "line number here"
 - indicate the filepath in "file": "relative/filepath"
 - if you put multiple lines in on line number, will replace only one line with multiple lines expanding the code.
6. if you need to add extra code at the bottom of the file or in an empty file, use add.
 - use "action": "add"
 - add code in "code": "put code here"
 - indicate the filepath in "file": "relative/filepath"

## Here is an example:

Command Results: already performed actions with their responses
User Prompt: "list the project folder"
Response:
{
    "action": "yuki",
    "command": "ls -ltra",
    "description": "listing files"
}

## Rules

* Run only one command at a time (do not use ";")
* If README.md doesnt exist create it and documen the thing.
* Check which files already exists.
* Create the missing files:
    - the main program file
    - the Makefile
    - the dependency installer script or requirements.txt in case of python
* You start directly in a folder, dont exit that folder.
* Visualize code with cat
* install dependencies
* dont use ../ in paths or absolute paths /a just relative some/path/here

## Commands

* "ls -ltra" : list the files in base folder
* "ls folder/" : lis another folder
* "touch file" : create and empty file
* "netstat -putan" : check the connections is like netstat -putan
* "ps auxf" : check the processes is like ps auxf
* "pwd" : check current path
* "add folder/file.py '\tdef main():'" : you can add a line of code to a file in this way
* "cat folder/file" : you can view content of a file
* "python3 file.py" : you can execute a python file
* "pip install module" : you can install modules
* "make" : you can make the project
* "file filename" : check which type of file is
* "binwalk filename" : check which formats contains the binary
* "r2 -q -c 'something'" : static reverse ingeneering a binary
* "curl -sk 'url'" : connect to a website
* "wget 'url'" : download a web resource

''' # system prompt inspired on decai.r2.js

def unimplemented():
    print('unimplemented!')
    exit(1)


def colorcat(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            code = file.read()
        lexer = get_lexer_for_filename(filename)
        print(highlight(code, lexer, TerminalFormatter()))
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no se encuentra.", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: No se puede decodificar '{filename}' como texto UTF-8.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def is_speech(audio_chunk, threshold=ENERGY_THRESHOLD):
    return np.max(np.abs(audio_chunk)) > threshold

def yuki_stt(duration=SILENCE_DURATION):
    print('Speak.')
    while True:
        audio_buffer = []
        silence_counter = 0
        max_chunks = int(MAX_DURATION / CHUNK_DURATION)
        silence_chunks = int(duration / CHUNK_DURATION)

        # Configurar stream de audio
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype="float32", blocksize=int(SAMPLERATE * CHUNK_DURATION)) as stream:
            for _ in range(max_chunks):
                # Leer fragmento de audio
                audio_chunk, _ = stream.read(int(SAMPLERATE * CHUNK_DURATION))
                audio_buffer.append(audio_chunk.flatten())

                # Verificar si hay voz
                if is_speech(audio_chunk):
                    silence_counter = 0  # Reiniciar contador si hay voz
                else:
                    silence_counter += 1  # Incrementar contador si hay silencio

                # Detener si se detecta suficiente silencio
                if silence_counter >= silence_chunks:
                    #print("Silence detected, stopping recording...")
                    break
        audio = np.concatenate(audio_buffer) if audio_buffer else np.array([])
        if len(audio) < SAMPLERATE * 0.5:  # Mínimo 0.5 segundos
            msg = "continue please"
            print(msg)
            yuki_tts(msg)
            continue
        print("Parsing...")
        result = model.transcribe(audio, language="en")
        txt = result["text"].strip()
        if txt:
            return txt


def yuki_stt_test():
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            model = whisper.load_model('turbo')
            result = model.transcribe('audio.mp3', language='en')
            print(result['text'])

def yuki_tts(text):
    text = text.replace('**','').replace('#','')
    text = re.sub(r'```[\s\S]*?```', '', text)
    print(text)

    try:
        #tts = TTS("tts_models/en/blizzard2013/capacitron-t2-c150_v2", progress_bar=False, gpu=False) # retry!
        #tts = TTS("tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)
        #tts = TTS("tts_models/en/vctk/vits", progress_bar=False, gpu=False)  # buena
        #tts = TTS("voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=False)
        #tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)  # Voz femenina clara
        #tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
        #tts = TTS("tts_models/en/ljspeech/vits", progress_bar=False, gpu=False) #muy buena
        tts = TTS(VOICE,  progress_bar=False, gpu=False)
        if not tts:
            print('fallo el tts')
            exit()
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                if tts.speakers:
                    audio = tts.tts(text, speaker=tts.speakers[0]) #, language="en")
                else:
                    audio = tts.tts(text) #, language="en")
        audio = np.array(audio, dtype=np.float32)
        sd.play(audio, samplerate=tts.synthesizer.output_sample_rate)
        sd.wait()
    except:
        print('failed -> ', text)



def AI(msg):
    print('Thinking ...')
    response = ollama.chat(model=MODEL, messages=[
      {
          'role':'system',
          'content': SYSTEM_PROMPT
      },
      {
        'role': 'user',
        'content': msg
      },
    ])
    return response['message']['content']

def prompt():
    p = ['user said:']
    while True:
        x = input('Ask (say go to process) => ').strip()
        if x == 'go':
            break
        p.append(x)
    return '\n'.join(p)

def confirm(cmd):
    global context

    if GO_AHEAD:
        context += '**user**\ngo ahead!\n'
        return True

    for i in range(3):
        print(cmd)
        print('run the command (yes/no/edit)? ')
        yuki_tts('run the command?')
        ans = yuki_stt(3).lower()
        if 'edit' in ans:
            user = yuki_stt()
            context += '**user adds this comment**\n' + user + '\n'
        elif 'yes' in ans:
            return True
        elif 'no' in ans:
            return False
        yuki_tts('sorry, can you repeat?')
    yuki_tts('sorry, i didnt understood.')
    exit()


def do_cd(cmd):
    path = cmd.split(' ')[1]
    if os.path.isdir(path):
        try:
            os.chdir(path)
            return True
        except:
            yuki_tts('permission denied')
    else:
        yuki_tts('sorry, cannot access to a path that is not a folder.')
    return False



def process_command(cmd):
    global context
    context += 'command:\n' + cmd + '\n'
    if confirm(cmd):
        yuki_tts('launching command')
        proc = subprocess.Popen(cmd, shell=True)
        stdout, stderr = proc.communicate()
        context += '**stdout**\n'
        if stdout:
            print(stdout)
            context += stdout + '\n---\n'
        else:
            context += 'there is no stdout.\n'
        if stderr:
            print('error:\n',stderr)
            context += '**stderr**\n' + stderr + '\n'
        else:
            context += 'command executed ok.\n'
        return True
    else:
        return True


def test():
    for i in range(4):
        yuki_tts(f'hello i am {MODELS[i]["name"]}')


def sanitize(cmd):
    cmd = cmd.strip().replace('`','').replace('$(','')
    spl = cmd.split(' ')
    for s in spl:
        if s == 'rm':
            print(f'rm blocked! cmd: {cmd}')
            return None
        elif s == 'cd':
            print(f'cd blocked! cmd: {cmd}')
            return None
        elif s.startswith('/'):
            print(f'absolute path blocked! cmd: {cmd}')
            return None
    return cmd

def lcode(code):
    code2 = ''
    spl = code.strip().split('\n')
    for i in range(len(spl)):
        code2 += f'{i+1}\t{spl[i]}\n'
    return code2


def edit(file, line, code):
    global context

    if file.startswith('..') or file.startswith('/'):
        print(f'alert: add bloqued with not alowed file-path: {file}')
        context += '\nremember: use always relative paths, {file} is absolute.\n'
        return

    line = int(line)
    if not os.path.isfile(file):
        add(file, code)
    blob = open(file,'r').read().strip()
    blob = blob.split('\n')
    if line > len(blob):
        add(file,code)
        context += 'edit: doing add instead of edit\n'
        return
    for i in range(len(blob)):
        if i+1 == line:
            blob[i] = code
            break
    try:
        code = '\n'.join(blob)
        open(file,'w').write(code)
    except Exception as e:
        context += '**edit error**\n' + str(e)
        return
    colorcat(file)
    context += f'file: {file}\n'
    context += lcode(code) + '#eof\n\n'

def add(file, code):
    if file.startswith('..') or file.startswith('/'):
        print(f'alert: bloqued this add: {file}')
        return

    global context
    try:
        open(file,'a').write(code)
        code = open(file,'r').read()
        colorcat(file)
        context += f'file {file}\n{code}\n\n'
    except Exception as e:
        context += '**edit error**\n' + str(e)

def clean_json(data):
    if '{' not in data or '}' not in data:
        return ''

    data = data.replace('```json\n','').replace('```','')
    off = data.find('}\n\n{')
    if off >= 0:
        data = data[:off+1]
    if not data.startswith('{'):
        off = data.find('{')
        data = data[off:]

    return data

def main():
    global context
    yuki_tts('Hello, what do you want to do?')
    request = yuki_stt()
    print('user> '+request)
    context += '**user request**\n' + request + '\n'
    while True:
        step = AI(context).strip()
        step = clean_json(step)
        #start_off = step.find('{')
        #end_off = step.find('}')+1
        #step = step[start_off:end_off].strip()
        context += '**yuki action**\n' + step + '\n'
        print(step)
        try:
            step = json.loads(step)
        except:
            context += '\nbad format, answer with just one valid json\n'
            print('json doesnt compile:')
            print(step)
            continue
        if 'description' in step:
            desc = step['description']
            yuki_tts(desc)
        if step['action'] == 'reply':
            yuki_tts(step['response'])
        elif step['action'] == 'edit':
            if 'line' in step and 'code' in step and 'file' in step:
                line = int(step['line'])
                code = step['code']
                file = step['file']
                edit(file, line, code)
            else:
                context += '\njson is ok but there are missing line or code or file tags\n'
        elif step['action'] == 'add':
            if 'code' in step and 'file' in step:
                code = step['code']
                file = step['file']
                add(file, code)
            else:
                context += '\njson is ok but there are missing code or file tags\n'
        elif step['action'] == 'yuki':
            cmd = step['command']
            if cmd == 'edit':
                context += '\ndont use the edit command because doesnt exist, use the edit action instead.\n'
            else:
                cmd = sanitize(cmd)
                if cmd:
                    process_command(cmd)
                    if cmd.startswith('cat '):
                        fname = cmd.split(' ')[1]
                        colorcat(fname)

        elif step['action'] == 'done':
            yuki_tts('program done, do you agree? (yes to finish)')
            user = yuki_stt().lower()
            if 'yes' in user:
                open('ctx','w').write(context)
                exit(1)
            context += '** user disagree, program not finished yet **\n' + user + '\n'
        else:
            context += '\naction not implemented, use known ones\n'


main()

