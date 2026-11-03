/* Browser-driven recovery. Every motion stays with the existing PS controller. */
(() => {
  function failureNeedsStop(op,owner,operationId) {
    if(!op||typeof op.busy!=='boolean')return true;
    if(op.owner!==owner||op.operation_id!==operationId){
      return !(op.busy&&typeof op.owner==='string'&&typeof op.operation_id==='string');
    }
    return !(!op.busy&&['rejected','finished'].includes(op.state)&&
      op.result?.status==='rejected'&&op.result?.physical_motion_attempted===false&&
      op.board_ownership?.status==='idle');
  }
  function describe(message) {
    const reasons={
      reset_device_still_owned:'设备仍被进程占用，请先停止其任务并等待退出，未强制解锁。',
      reset_lock_owner_unknown:'设备锁的归属不明确，请在高级设置核对占用情况，未强制解锁。',
      reset_state_changed_retry_review:'重置期间工位记录发生变化，请重新核对现场。',
      neutral_return_last_command_uncertain:'上一条舵机命令的结果不明确，不能自动回初始；请先核对实际姿态。',
      neutral_return_requires_known_empty_prepare_or_neutral:'当前姿态不在已核验的回程起点，需要人工恢复到已知姿态后再重置。',
      neutral_return_no_known_pose:'没有可核对的机械臂起始姿态，不能自动回初始。',
      neutral_return_calibration_jaw_not_open:'夹爪仍处于标定闭合位置，请先按受控流程张开空爪，再回初始。',
      neutral_return_fixed_markers_missing:'回程所需定位标记被遮挡，请清空标记区域后重新检查。',
      neutral_return_fixed_board_moved:'工位或相机位置发生变化，需要重新核对定位。',
      neutral_return_history_changed:'回程期间任务记录变化，流程已停止。',
      operation_busy_or_closed:'原任务尚未完全退出，请等待停稳后重新操作。'
    };
    return Object.entries(reasons).find(([key])=>String(message).includes(key))?.[1]||String(message);
  }
  class Recovery {
    constructor({read,send,owner,id,progress=()=>{},wait=ms=>new Promise(r=>setTimeout(r,ms)),clock=()=>performance.now(),visible=()=>document.visibilityState==='visible'}) {
      Object.assign(this,{read,send,owner,id,progress,wait,clock,visible});
      this.cancelled=false;this.expected=null;this.prepared=false;
    }
    check() {
      if(this.cancelled)throw Error('重置已停止，请核对工位；没有继续解锁或回初始。');
      if(!this.visible())throw Error('页面已离开前台，重置流程中止；请核对工位状态。');
    }
    async current(expected,match=true) {
      this.check();const op=await this.read();this.check();
      if(!op||typeof op.busy!=='boolean')throw Error('控制状态不可读，重置未继续。');
      if(match&&(op.operation_id??null)!==expected)throw Error('工位任务已变化，重置已中止；请重新核对现场。');
      return op;
    }
    async command(action,expected,extra={}) {
      this.check();await this.current(expected);
      return this.send(action,{owner:this.owner,expected_operation_id:expected,...extra});
    }
    async prepare(expected) {
      this.expected=expected;let op=await this.current(expected);
      if(op.reset_supported!==true||op.home_supported!==true)throw Error('当前工位不支持完整重置，请打开高级工位设置。');
      if(op.busy||op.board_ownership?.status==='active') {
        this.progress('stopping','正在停止当前任务，等待控制器退出。');
        await this.command('stop',expected);
        const deadline=this.clock()+30000;
        do {
          op=await this.current(expected);
          if(!op.busy&&op.board_ownership?.status!=='active')break;
          if(this.clock()>=deadline)throw Error('停止尚未确认，未解锁；请核对现场后重新操作。');
          await this.wait(500);
        }while(true);
      }
      this.prepared=true;
      this.progress('confirmation','任务已停止。请核对机械臂停稳、夹爪无夹物，双手撤出回程范围。已核验的标定闭爪会先缓慢张开。');
      return op;
    }
    async run(confirmed) {
      if(!this.prepared||confirmed!==true)throw Error('请先确认机械臂停稳、夹爪空且现场已清空。');
      let op=await this.current(this.expected);
      if(op.busy||op.board_ownership?.status==='active')throw Error('工位仍被占用，未解锁。');
      this.progress('unlocking','正在归档未决状态、清理停止标记并解锁工位。');
      const reset=await this.command('reset',this.expected,{request_id:this.id(),ready:true,reset_confirmed:true});
      if(reset?.state!=='idle'||reset?.result?.status!=='state_reset'||reset.operation_id!=null)throw Error('解锁结果未确认，未发送回初始动作；不会自动重试。');
      this.expected=null;
      op=await this.current(null);
      if(op.busy||op.board_ownership?.status!=='idle'||op.blockers?.length)throw Error('工位尚未就绪：'+(op.blockers?.join('；')||'控制权未释放'));
      this.progress('homing','正在按工位已核验的路线回初始位置，请保持页面在前台。');
      const homeId=this.id();this.expected=homeId;
      const reply=await this.command('home',null,{request_id:homeId,ready:true,home_confirmed:true});
      if(reply?.operation_id!==homeId||reply?.owner!==this.owner)throw Error('回初始请求未确认，不会再次提交；请核对工位。');
      const deadline=this.clock()+90000;
      while(true) {
        op=await this.current(homeId);
        if(op.owner!==this.owner)throw Error('回初始任务控制会话已变化，流程中止。');
        if(!op.busy)break;
        if(this.clock()>=deadline)throw Error('回初始仍未结束，请使用停止并核对工位；未自动重置后续状态。');
        await this.wait(500);
      }
      if(op.state!=='finished'||op.result?.status!=='neutral_return_command_completed')throw Error('回初始未完成：'+(op.result?.reason||op.reason||op.result?.status||op.state));
      this.progress('finishing','回初始命令已完整回显并等待结束，正在清理本轮操作状态。');
      const final=await this.command('reset',homeId,{request_id:this.id(),ready:true,reset_confirmed:true});
      if(final?.state!=='idle'||final?.result?.status!=='state_reset'||final.operation_id!=null)throw Error('最终空闲状态未确认，请核对工位；没有自动重试。');
      this.expected=null;
      op=await this.current(null);
      if(op.busy||op.board_ownership?.status!=='idle')throw Error('工位控制权仍未释放，暂不能开始下一次。');
      this.progress('position','工位已解锁，回初始命令已完成。请核对实际位置和现场，再准备下一次任务。');
      return op;
    }
    async verifyReady() {
      const op=await this.current(null);
      if(op.busy||op.board_ownership?.status!=='idle'||op.blockers?.length)throw Error('工位尚未就绪：'+(op.blockers?.join('；')||'控制权未释放'));
      return op;
    }
  }
  if(typeof module==='object'&&module.exports)module.exports={Recovery,describe,failureNeedsStop};
  else window.StationResetFlow={Recovery,describe,failureNeedsStop};
})();
