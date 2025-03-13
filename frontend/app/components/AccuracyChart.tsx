"use client";
import React from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

const AccuracyChart = () => {
  const chartData = [
    { month: "January", accuracy: 90 },
    { month: "February", accuracy: 97.2 },
    { month: "March", accuracy: 81.7 },
    { month: "April", accuracy: 99 },
    { month: "May", accuracy: 99.6 },
    { month: "June", accuracy: 85.3 },
    { month: "January", accuracy: 90 },
    { month: "February", accuracy: 97.2 },
    { month: "March", accuracy: 81.7 },
    { month: "April", accuracy: 38 },
    { month: "May", accuracy: 94.6 },
    { month: "June", accuracy: 85.3 },
  ];
  const chartConfig = {
    accuracy: {
      label: "accuracy",
      color: "#152d64",
    },
  } satisfies ChartConfig;

  return (
    <ChartContainer className="h-[300px] flex w-full" config={chartConfig}>
      <AreaChart
        height={50}
        accessibilityLayer
        data={chartData}
        margin={{
          left: 12,
          right: 12,
        }}
      >
        <CartesianGrid vertical={false} />
        <YAxis
          domain={[0, 100]} // Ensures values between 0 and 100
          tickCount={6} // Adjusts number of ticks (optional)
          tickMargin={8}
        />
        <XAxis
          dataKey="month"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          tickFormatter={(value) => value.slice(0, 3)}
        />
        <ChartTooltip
          cursor={false}
          content={<ChartTooltipContent indicator="line" />}
        />
        <Area
          dataKey="accuracy"
          type="natural"
          fill="#1358BD"
          fillOpacity={0.4}
          stroke="#1358BD"
          strokeWidth={3}
        />
      </AreaChart>
    </ChartContainer>
  );
};

export default AccuracyChart;
