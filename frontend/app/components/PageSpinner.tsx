import React from "react";
import { TailSpin } from "react-loader-spinner";

const PageSpinner = ({ color = "black" }: { color?: string }) => {
  return <TailSpin color={color} width={50} height={50} />;
};

export default PageSpinner;
