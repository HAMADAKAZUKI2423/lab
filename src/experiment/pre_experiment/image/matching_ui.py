"""前景窓内の操作UI。←/→で調整し、↓で一致を保存して次へ進む。"""
import tkinter as tk

class MatchingControls:
    def __init__(self,app):
        self.app=app;self.window=app.root
        self.frame=tk.Frame(self.window,bg='black',padx=10,pady=8)
        trial=app.current_trial
        tk.Label(self.frame,text=f"Trial {trial.trial_order}/{len(app.trial_list)}",bg='black',fg='gray').pack()
        tk.Label(self.frame,text='下の前景の見えやすさを、上と同じに調整\n背景の見えやすさではありません。',bg='black',fg='white').pack(pady=4)
        self.value=tk.DoubleVar(value=app.desired_gain)
        self.suppress=False
        self.slider=tk.Scale(self.frame,from_=0,to=1,resolution=app.session_config.gain_fine_step,
             showvalue=False,orient=tk.HORIZONTAL,length=350,variable=self.value,
             bg='black',fg='gray',highlightthickness=0,command=self.changed)
        self.slider.pack(fill='x',pady=5)
        tk.Label(self.frame,text='←/→：調整   Shift+←/→：微調整\n↓：決定',bg='black',fg='gray').pack()
        # 本試行は調整と↓確定のみ。評価・次へボタンは表示しない。
        self.buttons=[]
        commands={'<Left>':lambda e:self.step(-app.session_config.gain_step),
                  '<Right>':lambda e:self.step(app.session_config.gain_step),
                  '<Shift-Left>':lambda e:self.step(-app.session_config.gain_fine_step),
                  '<Shift-Right>':lambda e:self.step(app.session_config.gain_fine_step),
                  '<Down>':app.confirm_with_down}
        for key,command in commands.items():
            app.bind_control_key(key,command)
            # Scale/Buttonの標準矢印処理より先に処理し、二重変更を防ぐ。
            for widget in (self.slider,*self.buttons):
                widget.bind(key,lambda e,c=command:self.consume(c,e))
        app.place_control_frame(self.frame)
        self.slider.focus_set()
    @staticmethod
    def consume(command,event):
        command(event);return 'break'
    def changed(self,value):
        if not self.suppress: self.app.request_gain(float(value))
    def set(self,gain):
        self.suppress=True;self.value.set(gain);self.suppress=False
    def step(self,delta):
        gain=max(0.,min(1.,round(self.app.desired_gain+delta,8)))
        self.set(gain);self.app.request_gain(gain);return 'break'
    def disable(self):
        self.slider.configure(state='disabled')
        for b in self.buttons: b.configure(state='disabled')
    def destroy(self): self.frame.destroy()
