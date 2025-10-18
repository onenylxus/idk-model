from model import MistralBaseModel

if __name__ == "__main__":
    model = MistralBaseModel()
    prompt = "Hello, my name is"
    print(f"[Prompt] {prompt}\n")
    response = model.generate(prompt)
    print(f"[Response] {response}")
