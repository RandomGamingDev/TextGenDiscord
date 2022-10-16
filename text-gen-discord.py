import discord
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

TOKEN = "" # Put your token here
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

@client.event
async def on_ready():
    print("Client Started!\nLoading Server List...")
    print("SERVERS:\n--------")
    for guild in client.guilds:
        print(str(guild.name) + ': \n' + "ID: " + str(guild.id) + ", " + "COUNT: " + str(guild.member_count) + '\n')
    print("Loaded Server List!\nWe have logged in as {0.user}!".format(client))

@client.event
async def on_message(message):    
    if message.author == client.user:
        return

    if message.content[0:4] == "gen " and len(message.content) < 204:
        await message.channel.send("Generating...")
        sequence = (message.content[4:len(message.content)])
        inputs = tokenizer.encode(sequence, return_tensors='pt')
        outputs = model.generate(inputs, max_length=199, do_sample=True, temperature=1, top_k=50, pad_token_id=0)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        await message.channel.send("Generated!")
        await message.channel.send(text)

client.run(TOKEN)
