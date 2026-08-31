
ggplot(df_long,
       aes(x = Measure,
           y = factor(Phenotype, levels = rev(c(
             "High Adaptive",
             "Moderate Adaptive",
             "Behavioral Vulnerability",
             "High Risk"
           ))),
           fill = Mean)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = round(Mean, 1)),
            size = 4) +
  scale_fill_gradient(
    low = "#F7FBFF",
    high = "#08306B",
    name = "Mean"
  ) +
  labs(
    title = "Behavioral Phenotypes Across SDQ, ASBI, and CBRS",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      size = 16
    ),
    panel.grid = element_blank()
  )