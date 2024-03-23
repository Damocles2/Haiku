import tkinter
import customtkinter
import tkinter.font 

import torch
from transformers import pipeline

def delete_before_last_instance(input_string, word):
    last_index = input_string.rfind(word)
    if last_index != -1:  # Check if the word is found
        return input_string[last_index + len(word):]
    else: 
        return input_string


pipe = pipeline("text-generation", model="TheBloke/zephyr-7B-beta-AWQ", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {"role": "system","content": "You are a helpful assistant"},
]
global el_cum
el_cum = ""

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def clear_messages(entry, textbox, messages):
    global el_cum
    messages.clear()
    el_cum = ""
    textbox.configure(state="normal")
    textbox.delete('1.0', "end")
    textbox.configure(state="disabled")


def buttonpress_event(entry, textbox,messages):
    global el_cum
    userprompt = {"role": "user", "content": entry.get()}
    el_cum += ("User: \n" + entry.get() + "\n\n")
    textbox.configure(state="normal")
    textbox.delete('1.0', "end")
    textbox.insert("end", el_cum)  # Use "end" to insert text at the end of the textbox
    textbox.see("end")
    textbox.configure(state="disabled")
    textbox.update()
    entry.delete(0, 'end')
    entry.update()
    
    messages.append(userprompt.copy())
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=2048, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
    response_text = delete_before_last_instance(outputs[0]["generated_text"], "<|assistant|>")
    el_cum += ("Assistant:" + (delete_before_last_instance(outputs[0]["generated_text"], "<|assistant|>")) + "\n\n")

    response_to_append = {"role": "assistant", "content": response_text}
    textbox.configure(state="normal")
    textbox.delete('1.0', "end")
    textbox.insert("end", el_cum)  # Use "end" to insert text at the end of the textbox
    textbox.see("end")
    textbox.configure(state="disabled")



class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Haiku")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure((1), weight=0)
        self.grid_rowconfigure((0), weight=1)
        self.grid_rowconfigure((1), weight=0)

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="Prompt")
        self.entry.grid(row=1, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=lambda: buttonpress_event(self.entry, self.textbox,messages), text="Send")
        self.main_button_1.grid(row=1, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self)
        self.textbox.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew", columnspan=3, state="disabled")

        self.textbox.configure(font=("Noto Sans", 14, "bold"), wrap="word")
        self.entry.bind("<Return>", lambda event: buttonpress_event(self.entry, self.textbox, messages))
        self.bind_all("<Control-d>", lambda event: clear_messages(self.entry, self.textbox, messages))
        self.iconbitmap("opt.ico")



if __name__ == "__main__":
    app = App()
    app.mainloop()
