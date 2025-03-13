import * as React from "react";
import { SVGProps } from "react";
const SelectedFilterIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg width={24} height={24} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <rect x={1} y={1} width={22} height={22} rx={4} fill="#0066FF" />
    <rect x={1} y={1} width={22} height={22} rx={4} stroke="#0066FF" strokeWidth={2} />
    <path d="M16 9L10.5 14.5L8 12" stroke="white" strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
export default SelectedFilterIcon;