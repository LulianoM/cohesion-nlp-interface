import openai

class OpenAI:

    def connect_openai(data, key):
        openai.api_key = key

        prompt = "Com o texto a seguir avalie entre 1 a 5 a coesão do texto."+data

        try:
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=prompt,
                temperature=0.7,
                max_tokens=545,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0.58,
                stop=["###"])

            return dict(response["choices"][0])["text"].replace("\n\n", "") , 200
        except Exception as err:
            return {"Error": err}, 500