import json
import re

fileName = '/home/simon/workspace/rag-stripe/ollama-api.md'

fileContent = open(fileName).read().splitlines()
currentHeader = []
chunks = []

def parseHeader(header: list[str]):
    currentHeader = ''

    for index, h in enumerate(header):
        if(index != 0):
            currentHeader += ' > '
        currentHeader += h.lstrip('# ')
    return currentHeader

currentChunk = {'heading': '', 'text': '', 'index': 0}
currentChunkIdx = 0

for f in fileContent:
    ## Request + Response shouldnt be standalone chunks so add them to the chunk before
    if f.startswith('#') and f.lstrip('# ').lower() not in ('request', 'response'):
        if(len(currentHeader) == 0):
                ## init - dont want to add first line as chapter, if something was before the chapter
              currentHeader.append(f)
        else:
            if len(currentChunk['text'].strip()) > 0:
                ## if we get to another chunk start - we found another # chapter
                # save the prev one to export variable
                currentChunk['text'] = parseHeader(currentHeader) + '\n' + currentChunk['text']
                currentChunkIdx = currentChunkIdx + 1
                chunks.append(currentChunk)

            currentChunk = {'heading': '', 'text': '', 'index': 0}
            headerLen = len(f) - len(f.lstrip('#'))
            if headerLen == len(currentHeader):
                currentHeader.pop()
                currentHeader.append(f)
            elif headerLen > len(currentHeader):
                currentHeader.append(f)
            elif headerLen < len(currentHeader):
                while headerLen < len(currentHeader):
                    currentHeader.pop()
                currentHeader.pop()
                currentHeader.append(f)

        currentChunk['heading'] = parseHeader(currentHeader)
    else:
        ## replace all amount of spaces with just one
        currentChunk['text'] += re.sub(r"\s+", " ", f.lstrip('#')) + '\n'

    currentChunk['index'] = currentChunkIdx


## append the last chunk
currentChunk['text'] = parseHeader(currentHeader) + '\n' + currentChunk['text']
chunks.append(currentChunk)

np.save("ollamaVectors.npy", np.array(vectors))   # save
