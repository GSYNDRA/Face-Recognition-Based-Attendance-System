tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: Input name").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                  text='Input',
                  command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)
