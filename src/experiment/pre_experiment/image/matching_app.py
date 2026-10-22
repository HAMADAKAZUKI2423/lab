"""旧ratingアプリを保持し、matchingのみの位置合わせ・提示・保存を追加。"""
import math, random, time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk
from .app import ImageExperimentApp, WIN1_MARKER_COLOR, WIN2_MARKER_COLOR
from .calibration import apply_trial_calibration
from .conditions import get_condition
from .stimuli import participant_seed, condition_order_for_participant, build_trials
from .matching_layout import (Box,make_layout,validate_layout,layout_metadata,draw_brackets)
from .matching_stimuli import (calibration_files_metadata,validate_calibration,prepare_matching_stimulus)
from .matching_results import (initial_gain_plan,save_matching_manifest,save_matching_session,build_matching_row)
from .matching_ui import MatchingControls

class ImageMatchingApp(ImageExperimentApp):
    def __init__(self,root,session_config):
        if session_config.task_mode!='matching': raise ValueError('Matching configuration required')
        self.calibration_file_info=calibration_files_metadata(session_config.display_dir)
        # 本試行・校正・調整履歴をメモリに保持し、途中ではファイルを作らない。
        self.defer_result_saving=True
        self.controls=None;self.confirm_down_held=False
        self.matching_state='setup';self.preflight_complete=False;self.preflight_records=[]
        self.adjustment_history=[];self.initial_gains={};self.pending_render=None
        self.pending_transition=None;self.matching_prepared=None;self.trial_started=None
        self.frozen_viewports=None;self.desired_gain=0.;self.rendered_gain=0.;self.adjustment_count=0
        super().__init__(root,session_config)
        validate_calibration(self.defocus_display_calibration)
        self.root.title('Image Matching - Foreground plane')
        self.win1.title('Image Matching - Background plane')
        self.root.protocol('WM_DELETE_WINDOW',self.request_close)
        self.win1.protocol('WM_DELETE_WINDOW',self.request_close)

    def _setup_windows(self):
        # 前景(root)と背景(win1)の2窓のみ。操作Frameは前景窓内へplaceする。
        super()._setup_windows()
        # 試行間でキーバインドを解除しても、Downの離鍵は追跡する。
        self.root.bind('<KeyRelease-Down>',self.release_confirm_down)
        self.root.bind('<Escape>',lambda event:self.request_close())

    def start_calibration_sequence(self):
        # 親rating版と異なり、保存先名を決めるだけ。mkdirは全試行完了後。
        self.win1.update_idletasks()
        self.width=self.win1.winfo_width();self.height=self.win1.winfo_height()
        participant=self.participant_id.get().strip() or 'participant'
        safe_id=''.join(c if c.isalnum() or c in '-_' else '_' for c in participant)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.result_dir=self.session_config.result_root / f'{safe_id}_{stamp}'
        self.current_calib_eye_idx=0;self.calib_results={};self.detailed_defocus_results=[]
        self.current_trial_index=0;self.active_block_id=0;self.results=[]
        self.adjustment_history=[];self.matching_state='setup'
        self.start_eye_calibration()

    def bind_control_key(self,key,command):
        def callback(event): command(event);return 'break'
        self.key_bindings[key]=self.root.bind(key,callback)

    def release_confirm_down(self,event=None):
        self.confirm_down_held=False
        return 'break'

    def confirm_with_down(self,event=None):
        # 押しっぱなしのキーリピートで次試行まで連続確定しない。
        if self.matching_state!='adjusting' or self.confirm_down_held: return 'break'
        self.confirm_down_held=True
        return self.confirm_matching('matched')

    def place_control_frame(self,frame,*,calibration=False):
        """Canvasサイズを変えず、刺激領域の外に操作Frameを配置する。"""
        self.root.update_idletasks()
        layout=self.layout_for_current_eye() if calibration else self.current_layout
        names=('test',) if calibration else ('reference_singleplane','singleplane' if self.matching_prepared.background_xyz is not None else 'test')
        fg,bg=self.viewports()
        occupied=[layout[name].translated(fg[2],fg[3]) for name in names]
        # Referenceは前景面だけに提示し、背景面の上段は黒のまま。
        if calibration or self.matching_prepared.physical_background is not None:
            occupied.append(layout['background'].translated(bg[2],bg[3]))
        width,height=frame.winfo_reqwidth(),frame.winfo_reqheight()
        pad=12
        xs=[pad,fg[0]-width-pad]
        ys=[pad,fg[1]-height-pad]
        # 四隅だけでなく、刺激端に接する空き領域も候補にする。
        for box in occupied:
            xs.extend((box.x1-fg[2]+pad,box.x0-fg[2]-width-pad))
            ys.extend((box.y1-fg[3]+pad,box.y0-fg[3]-height-pad))
        for y in dict.fromkeys(ys):
            for x in dict.fromkeys(xs):
                local=Box(x,y,x+width,y+height)
                if not local.contains(fg[0],fg[1]): continue
                absolute=local.translated(fg[2],fg[3])
                if any(absolute.intersects(box) for box in occupied): continue
                frame.place(x=x+fg[2]-self.root.winfo_rootx(),
                            y=y+fg[3]-self.root.winfo_rooty(),anchor='nw')
                frame.lift()
                self.root.update_idletasks()
                return
        raise ValueError('No free area for controls inside the foreground display. Stimuli were not moved or resized.')

    def viewports(self):
        return ((self.canvas2.winfo_width(),self.canvas2.winfo_height(),self.canvas2.winfo_rootx(),self.canvas2.winfo_rooty()),
                (self.canvas1.winfo_width(),self.canvas1.winfo_height(),self.canvas1.winfo_rootx(),self.canvas1.winfo_rooty()))

    def layout_for_current_eye(self):
        fg,bg=self.viewports()
        return make_layout(self.session_config,fg[:2],bg[:2],self.offset_x.get(),self.offset_y.get())

    def update_calibration_view(self,*args):
        self.root.update_idletasks()
        self.canvas1.delete('calib');self.canvas2.delete('calib')
        layout=self.layout_for_current_eye()
        for name in ('background','background_square'):
            draw_brackets(self.canvas1,layout[name],WIN1_MARKER_COLOR,7.5)
        draw_brackets(self.canvas2,layout['test'],WIN2_MARKER_COLOR)
        x,y=layout['test'].center
        self.canvas2.create_line(x-12,y,x+12,y,fill=WIN2_MARKER_COLOR,tags='calib')
        self.canvas2.create_line(x,y-12,x,y+12,fill=WIN2_MARKER_COLOR,tags='calib')

    def setup_calibration_ui(self):
        self._destroy_frame('ctrl_frame');self.clear_key_bindings();self.update_calibration_view()
        self.ctrl_frame=tk.Frame(self.root,bg='gray',padx=8,pady=8)
        eye=self.calibration_eyes[self.current_calib_eye_idx]
        tk.Label(self.ctrl_frame,text=f'{eye} eye: align the red frame with\nthe LOWER white square ({self.session_config.test_center_deg:+g} deg).\nArrow keys move the background frame.\nPress Enter when aligned.',bg='gray',fg='white',wraplength=350).pack(pady=10)
        tk.Button(self.ctrl_frame,text='Alignment done / Defocus matching',command=self.start_eye_defocus_matching).pack(pady=8)
        for key,dx,dy in (('<Left>',-1,0),('<Right>',1,0),('<Up>',0,-1),('<Down>',0,1)):
            self.bind_control_key(key,lambda e,x=dx,y=dy:self.adjust_offset(x,y))
        self.bind_control_key('<Return>',lambda e:self.start_eye_defocus_matching())
        self.place_control_frame(self.ctrl_frame,calibration=True)
        self.root.focus_set()

    def start_eye_defocus_matching(self):
        super().start_eye_defocus_matching()

    def begin_experiment(self,event=None):
        if self.matching_state!='setup': return
        self.root.update_idletasks();self.matching_state='preflight'
        self._destroy_frame('ctrl_frame');self.clear_key_bindings()
        self.participant_seed=participant_seed(self.session_config.random_seed,self.participant_id.get().strip())
        self.condition_order=condition_order_for_participant(self.session_config.conditions,self.participant_id.get(),self.session_config.random_seed)
        self.trial_list=build_trials(self.selected_pairs,self.condition_order,random.Random(self.participant_seed))
        if len(self.trial_list)!=self.session_config.expected_trial_count: raise RuntimeError('Trial count mismatch')
        self.initial_gains=initial_gain_plan(self.trial_list,self.participant_seed)
        self.frozen_viewports=self.viewports();self.preflight_index=0
        self.canvas1.delete('all');self.canvas2.delete('all')
        self.ctrl_frame=tk.Frame(self.root,bg='gray');self.ctrl_frame.place(relx=.5,rely=.5,anchor='center')
        self.progress_label=tk.Label(self.ctrl_frame,text='Checking geometry, calibration and gamut...',bg='black',fg='white');self.progress_label.pack(pady=30)
        # Manifestも結果も全試行完了後に保存する。
        self.pending_transition=self.root.after(1,self._preflight_next)

    def prepare(self,trial):
        if self.viewports()!=self.frozen_viewports: raise ValueError('Display window moved/resized. Restart calibration; positions were not silently updated.')
        apply_trial_calibration(self,trial.condition_id)
        layout=self.layout_for_current_eye();fg,bg=self.viewports()
        validate_layout(layout,fg[:2],bg[:2],get_condition(trial.condition_id).plane_mode=='single')
        right=self.calib_results['Right'];left=self.calib_results['Left']
        prepared=prepare_matching_stimulus(trial,self.session_config,layout=layout,
            calibration=self.defocus_display_calibration,dominant_eye=self.participant_dominance.get(),
            left_pupil_mm=float(left['pd_mean']),right_pupil_mm=float(right['pd_mean']),ipd_mm=float(self.participant_ipd.get()))
        return layout,prepared

    def _preflight_next(self):
        self.pending_transition=None
        if self.matching_state!='preflight': return
        if self.preflight_index==len(self.trial_list):
            self.preflight_complete=True;self.matching_state='ready'
            self._destroy_frame('ctrl_frame')
            self.current_trial_index=0;self.active_block_id=0
            messagebox.showinfo('Matching instructions','Same viewing eye for upper reference and lower test.\nMatch the FOREGROUND features, not brightness alone.\nThe upper reference is fixed; only lower foreground light changes.\nUse the slider or Left/Right keys; press Down to confirm and advance.\nControls are inside the foreground window, outside both images.\nXYZ-increment calibration: black-inclusive luminance must be verified on hardware.',parent=self.root)
            self.run_trial();return
        trial=self.trial_list[self.preflight_index]
        try:
            layout,prepared=self.prepare(trial)
            self.preflight_records.append({'Trial_ID':trial.trial_order,'Condition':trial.condition_id,
                'Pair_ID':trial.pair.pair_id,'metadata':dict(prepared.metadata),'layout':layout_metadata(layout)})
            self.preflight_index+=1
            self.progress_label.configure(text=f'Preflight {self.preflight_index}/{len(self.trial_list)}')
            self.pending_transition=self.root.after(1,self._preflight_next)
        except Exception as exc: self.fail_matching(exc)

    def run_trial(self):
        if self.current_trial_index>=len(self.trial_list): self.finish_experiment();return
        trial=self.trial_list[self.current_trial_index]
        if trial.block_id!=self.active_block_id:
            self.matching_state='block_confirmation';self.show_block_confirmation(trial);return
        self.matching_state='preparing';self.current_trial=trial
        try:
            self.current_layout,self.matching_prepared=self.prepare(trial)
            self.current_layout_metadata=layout_metadata(self.current_layout)
            self.current_prepared=None
            self.canvas1.delete('all');self.canvas2.delete('all')
            self.photo_reference=ImageTk.PhotoImage(self.matching_prepared.reference_image)
            rb=self.current_layout['reference_singleplane']
            self.canvas2.create_image(rb.x0,rb.y0,
                image=self.photo_reference,anchor='nw',tags='matching')
            # Referenceは前景7:背景3のαブレンド画像。背景面上段には描画しない。
            if self.matching_prepared.physical_background is not None:
                self.photo_background=ImageTk.PhotoImage(self.matching_prepared.physical_background)
                b=self.current_layout['background'];self.canvas1.create_image(b.x0,b.y0,image=self.photo_background,anchor='nw',tags='matching')
            self.test_item=None;self.adjustment_count=0;self.trial_started=None
            self.desired_gain=self.initial_gains[trial.trial_order]
            self.render_now(log=False)
            if self.matching_state=='error': return
            self.controls=MatchingControls(self);self.root.update_idletasks()
            self.assert_controls_outside_stimuli()
            self.matching_state='adjusting';self.trial_started=time.monotonic();self.log_adjustment('initial')
        except Exception as exc: self.fail_matching(exc)

    def assert_controls_outside_stimuli(self):
        window=self.controls.frame
        control=Box(window.winfo_rootx(),window.winfo_rooty(),window.winfo_rootx()+window.winfo_width(),window.winfo_rooty()+window.winfo_height())
        fg,bg=self.viewports()
        boxes=[self.current_layout['reference_singleplane'].translated(fg[2],fg[3]),
               self.current_layout['singleplane' if self.matching_prepared.background_xyz is not None else 'test'].translated(fg[2],fg[3])]
        if self.matching_prepared.physical_background is not None: boxes.append(self.current_layout['background'].translated(bg[2],bg[3]))
        if not Box(control.x0-fg[2],control.y0-fg[3],control.x1-fg[2],control.y1-fg[3]).contains(fg[0],fg[1]):
            raise ValueError('Controls are outside the foreground display')
        if any(control.intersects(b) for b in boxes):
            raise ValueError('In-window controls overlap a stimulus. Stimulus positions were not changed.')

    def log_adjustment(self,event):
        trial=self.current_trial
        self.adjustment_history.append({'ID':self.participant_id.get(),'Trial_ID':trial.trial_order,
          'Condition':trial.condition_id,'Pair_ID':trial.pair.pair_id,
          'Elapsed_s':0. if self.trial_started is None else time.monotonic()-self.trial_started,
          'Requested_Gain':self.desired_gain,'Displayed_Gain':self.rendered_gain,'Event':event})

    def request_gain(self,gain):
        if self.matching_state!='adjusting': return
        if not math.isfinite(gain): self.fail_matching(ValueError('Non-finite gain'));return
        gain=max(0.,min(1.,gain))
        if gain==self.desired_gain: return
        self.desired_gain=gain;self.adjustment_count+=1
        if self.pending_render is None:
            self.pending_render=self.root.after(self.session_config.render_interval_ms,self.render_now)

    def render_now(self,log=True):
        self.pending_render=None
        if self.matching_state not in {'preparing','adjusting','confirming'}: return
        try:
            if self.viewports()!=self.frozen_viewports: raise ValueError('Display geometry changed during matching')
            image,self.render_metrics=self.matching_prepared.render(self.desired_gain)
            self.photo_test=ImageTk.PhotoImage(image)
            box=self.current_layout['singleplane' if self.matching_prepared.background_xyz is not None else 'test']
            if self.test_item is None:
                self.test_item=self.canvas2.create_image(box.x0,box.y0,image=self.photo_test,anchor='nw',tags='matching')
            else: self.canvas2.itemconfigure(self.test_item,image=self.photo_test)
            self.rendered_gain=self.desired_gain
            self.root.update_idletasks()
            if log and self.trial_started is not None: self.log_adjustment('render')
        except Exception as exc: self.fail_matching(exc)

    def flush_render(self):
        if self.pending_render is not None: self.root.after_cancel(self.pending_render);self.pending_render=None
        self.render_now()

    def confirm_matching(self,status):
        if self.matching_state!='adjusting': return 'break'
        if status not in {'matched','above_range','below_range','cannot_judge'}: raise ValueError(status)
        endpoint={'above_range':1.,'below_range':0.}.get(status)
        if endpoint is not None and self.desired_gain!=endpoint:
            self.controls.set(endpoint);self.request_gain(endpoint);self.flush_render()
            messagebox.showinfo('Check endpoint','The test is now at the corresponding limit. Observe it before confirming again.',parent=self.root)
            return 'break'
        self.request_gain(float(self.controls.value.get()))
        self.flush_render()
        if self.matching_state!='adjusting': return 'break'
        try: self.assert_controls_outside_stimuli()
        except Exception as exc: self.fail_matching(exc);return 'break'
        self.matching_state='confirming';self.controls.disable()
        elapsed=time.monotonic()-self.trial_started
        self.log_adjustment('confirm:'+status)
        self.results.append(build_matching_row(self,status,elapsed))
        self.matching_state='between_trials'
        self.controls.destroy();self.controls=None;self.clear_key_bindings()
        self.canvas1.delete('all');self.canvas2.delete('all');self.current_trial_index+=1
        if self.current_trial_index>=len(self.trial_list): callback=self.finish_experiment
        else:
            next_trial=self.trial_list[self.current_trial_index]
            callback=self.start_break if (next_trial.block_id==self.active_block_id and next_trial.within_block_order==self.session_config.trials_before_break+1) else self.run_trial
        self.pending_transition=self.root.after(500,callback)
        return 'break'

    def _checkpoint_results(self):
        # 既存呼出しとの互換用。完了以外は絶対に保存しない。
        if self.matching_state!='finished': return None
        return save_matching_session(self)

    def fail_matching(self,exc):
        self.matching_state='error';self.cancel_pending();self.clear_key_bindings()
        if self.controls is not None: self.controls.disable()
        self.canvas1.delete('all');self.canvas2.delete('all')
        messagebox.showerror('Matching stopped',str(exc)+'\nIncomplete session results were not saved. No geometry, background or luminance limits were silently changed. Correct the setup and restart.',parent=self.root)

    def cancel_pending(self):
        for name in ('pending_render','pending_transition'):
            identity=getattr(self,name,None)
            if identity is not None:
                self.root.after_cancel(identity);setattr(self,name,None)

    def request_close(self):
        if self.matching_state=='finished':
            self.root.destroy();return
        if not messagebox.askyesno('Discard session',
                'Stop and discard this session?\nNo trial, adjustment or defocus results will be saved.',
                parent=self.root): return
        self.matching_state='aborted';self.cancel_pending();self.clear_key_bindings()
        # 中断試行をaborted行として追加しない。以前の完了セッションには触れない。
        self.results.clear();self.adjustment_history.clear();self.detailed_defocus_results.clear()
        self.root.destroy()

    def finish_experiment(self):
        if self.matching_state in {'finished','aborted','error'}: return
        self.cancel_pending();self.clear_key_bindings();self.matching_state='finished'
        try:
            output=self._checkpoint_results()
        except Exception as exc:
            # 全試行の回答はメモリに維持し、保存失敗時は再試行できるようにする。
            self.matching_state='save_error'
            self.canvas1.delete('all');self.canvas2.delete('all');self._destroy_frame('ctrl_frame')
            self.ctrl_frame=tk.Frame(self.root,bg='gray',padx=20,pady=20)
            self.ctrl_frame.place(relx=.5,rely=.5,anchor='center')
            tk.Label(self.ctrl_frame,text='Saving failed. All responses remain in memory.\nPress Enter to retry saving. Closing discards this session.',bg='gray',fg='white').pack()
            tk.Button(self.ctrl_frame,text='Retry saving',command=self.finish_experiment).pack(pady=10)
            self.bind_control_key('<Return>',lambda event:self.finish_experiment())
            messagebox.showerror('Save failed',str(exc)+'\nNo partial result session was published.',parent=self.root)
            return
        matched=sum(r['Match_Status']=='matched' for r in self.results)
        messagebox.showinfo('Finished',f'Matching complete: {len(self.results)} responses, {matched} matched.\n{output}',parent=self.root)
        self.root.destroy()
