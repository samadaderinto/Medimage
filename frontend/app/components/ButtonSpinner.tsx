import React from "react";
import { TailSpin } from "react-loader-spinner";

const ButtonSpinner = ({ color = "white" }: { color: string }) => {
  return <TailSpin color={color} width={20} height={20} />;
};

export default ButtonSpinner;
