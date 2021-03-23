class DMI_STRAT(Strategy):
  dmi_window = 7
  loss_percentage= 90
  adx_threshold = 0
    
  def init(self):
    super().init()
    self.dmi_neg, self.dmi_pos, self.dm_adx = self.I(DMI, self.data.df ,self.dmi_window)

  def next(self):
        
    if crossover(self.dmi_pos, self.dmi_neg):
      if self.position:
        self.position.close()
      if self.dm_adx[-1] > self.adx_threshold:
        self.buy(sl=self.data.Close[-1]*(1-(self.loss_percentage/100)))

    elif crossover(self.dmi_neg, self.dmi_pos):
      if self.position:
        self.position.close()
      if self.dm_adx[-1] > self.adx_threshold:
        self.sell(sl=self.data.Close[-1]*(1+(self.loss_percentage/100)))