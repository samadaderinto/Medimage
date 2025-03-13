import * as React from "react";
import { SVGProps } from "react";
const DownArrow = (props: SVGProps<SVGSVGElement>) => (
  <svg width={16} height={16} viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <path d="M4 6L8 10L12 6" stroke="#3745AC" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
export default DownArrow;