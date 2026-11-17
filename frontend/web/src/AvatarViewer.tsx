import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

function makeCapsule(
  radius: number,
  length: number,
  material: THREE.Material,
  position: THREE.Vector3,
): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(radius, length, 5, 10),
    material,
  );
  mesh.position.copy(position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildAvatar(): {
  group: THREE.Group;
  breathing: THREE.Group;
  head: THREE.Group;
  arms: THREE.Group[];
} {
  const group = new THREE.Group();
  const breathing = new THREE.Group();
  group.add(breathing);

  const skin = new THREE.MeshStandardMaterial({
    color: 0xc98d68,
    roughness: 0.72,
  });
  const shirt = new THREE.MeshStandardMaterial({
    color: 0x52785e,
    roughness: 0.68,
  });
  const shirtLight = new THREE.MeshStandardMaterial({
    color: 0x78966f,
    roughness: 0.72,
  });
  const trousers = new THREE.MeshStandardMaterial({
    color: 0x344b48,
    roughness: 0.82,
  });
  const shoes = new THREE.MeshStandardMaterial({
    color: 0x463d38,
    roughness: 0.82,
  });
  const hair = new THREE.MeshStandardMaterial({
    color: 0x292a27,
    roughness: 0.9,
  });
  const eyeWhite = new THREE.MeshStandardMaterial({
    color: 0xf6f1e7,
    roughness: 0.5,
  });
  const eyeDark = new THREE.MeshStandardMaterial({
    color: 0x312c29,
    roughness: 0.45,
  });

  const torso = makeCapsule(
    0.34,
    0.69,
    shirt,
    new THREE.Vector3(0, 1.32, 0),
  );
  torso.scale.set(1, 1.08, 0.8);
  breathing.add(torso);

  const collar = new THREE.Mesh(
    new THREE.TorusGeometry(0.15, 0.025, 8, 24),
    shirtLight,
  );
  collar.position.set(0, 1.82, 0.08);
  collar.rotation.x = Math.PI / 2;
  breathing.add(collar);

  const pelvis = new THREE.Mesh(
    new THREE.SphereGeometry(0.31, 24, 16),
    trousers,
  );
  pelvis.position.set(0, 0.83, 0);
  pelvis.scale.set(1, 0.55, 0.76);
  breathing.add(pelvis);

  for (const side of [-1, 1]) {
    const leg = makeCapsule(
      0.145,
      0.48,
      trousers,
      new THREE.Vector3(side * 0.16, 0.43, 0),
    );
    breathing.add(leg);

    const shoe = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 20, 14),
      shoes,
    );
    shoe.position.set(side * 0.16, 0.105, 0.075);
    shoe.scale.set(0.9, 0.55, 1.4);
    breathing.add(shoe);
  }

  const neck = new THREE.Mesh(
    new THREE.CylinderGeometry(0.11, 0.13, 0.22, 16),
    skin,
  );
  neck.position.set(0, 1.91, 0);
  breathing.add(neck);

  const head = new THREE.Group();
  head.position.set(0, 2.26, 0);
  breathing.add(head);
  const face = new THREE.Mesh(
    new THREE.SphereGeometry(0.31, 32, 24),
    skin,
  );
  face.scale.set(0.86, 1.08, 0.84);
  face.castShadow = true;
  head.add(face);

  const hairCap = new THREE.Mesh(
    new THREE.SphereGeometry(0.31, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.48),
    hair,
  );
  hairCap.position.set(0, 0.08, -0.025);
  hairCap.scale.set(1.02, 0.94, 1.04);
  head.add(hairCap);

  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(
      new THREE.SphereGeometry(0.035, 12, 10),
      eyeWhite,
    );
    eye.position.set(side * 0.105, 0.015, 0.239);
    eye.scale.set(1, 0.78, 0.45);
    head.add(eye);

    const pupil = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 10, 8),
      eyeDark,
    );
    pupil.position.set(side * 0.105, 0.014, 0.254);
    head.add(pupil);
  }

  const nose = new THREE.Mesh(
    new THREE.SphereGeometry(0.04, 12, 10),
    skin,
  );
  nose.position.set(0, -0.055, 0.267);
  nose.scale.set(0.76, 0.85, 0.65);
  head.add(nose);

  const arms: THREE.Group[] = [];
  for (const side of [-1, 1]) {
    const shoulder = new THREE.Group();
    shoulder.position.set(side * 0.37, 1.65, 0);
    shoulder.rotation.z = side * 0.3;
    breathing.add(shoulder);
    arms.push(shoulder);

    shoulder.add(
      makeCapsule(
        0.14,
        0.42,
        shirt,
        new THREE.Vector3(0, -0.27, 0),
      ),
    );

    const forearm = new THREE.Group();
    forearm.position.set(0, -0.51, 0.015);
    forearm.rotation.z = -side * 0.12;
    shoulder.add(forearm);
    forearm.add(
      makeCapsule(
        0.105,
        0.36,
        skin,
        new THREE.Vector3(0, -0.22, 0),
      ),
    );

    const palm = makeCapsule(
      0.095,
      0.13,
      skin,
      new THREE.Vector3(side * -0.015, -0.47, 0.015),
    );
    forearm.add(palm);

    for (let finger = 0; finger < 4; finger += 1) {
      const fingerMesh = makeCapsule(
        0.022,
        0.085,
        skin,
        new THREE.Vector3(
          side * -0.015 + (finger - 1.5) * 0.042,
          -0.575 - Math.sin((finger / 3) * Math.PI) * 0.018,
          0.02,
        ),
      );
      forearm.add(fingerMesh);
    }

    const thumb = makeCapsule(
      0.027,
      0.075,
      skin,
      new THREE.Vector3(side * 0.085, -0.49, 0.03),
    );
    thumb.rotation.z = -side * 0.75;
    forearm.add(thumb);
  }

  return { group, breathing, head, arms };
}

export default function AvatarViewer() {
  const mountRef = useRef<HTMLDivElement>(null);
  const avatarRef = useRef<THREE.Group | null>(null);
  const [renderError, setRenderError] = useState("");

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    } catch (error) {
      setRenderError(
        error instanceof Error
          ? `3D avatar could not start: ${error.message}`
          : "3D avatar could not start. Check browser WebGL support.",
      );
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 40);
    camera.position.set(0, 1.75, 6.2);
    camera.lookAt(0, 1.32, 0);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.domElement.className = "avatar-canvas";
    renderer.domElement.setAttribute("aria-label", "Rotatable 3D avatar preview");
    mount.appendChild(renderer.domElement);
    setRenderError("");

    scene.add(new THREE.HemisphereLight(0xfff5e8, 0x657963, 2.1));

    const keyLight = new THREE.DirectionalLight(0xfff0d7, 3.2);
    keyLight.position.set(-3, 5, 5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xc8e3d0, 1.8);
    fillLight.position.set(3, 3, 2);
    scene.add(fillLight);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(1.35, 48),
      new THREE.MeshStandardMaterial({
        color: 0xd8e4d3,
        roughness: 0.9,
        metalness: 0,
      }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0.018;
    floor.receiveShadow = true;
    scene.add(floor);

    const avatar = buildAvatar();
    scene.add(avatar.group);
    avatarRef.current = avatar.group;

    let animationFrame = 0;
    const animationStart = performance.now();
    const render = () => {
      const elapsed = (performance.now() - animationStart) / 1000;
      avatar.breathing.position.y = Math.sin(elapsed * 1.7) * 0.018;
      avatar.head.rotation.y = Math.sin(elapsed * 0.55) * 0.045;
      avatar.arms[0].rotation.z = -0.3 + Math.sin(elapsed * 1.1) * 0.018;
      avatar.arms[1].rotation.z = 0.3 - Math.sin(elapsed * 1.1) * 0.018;
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(render);
    };

    const resizeObserver = new ResizeObserver(() => {
      const { width, height } = mount.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    });
    resizeObserver.observe(mount);
    const initialBounds = mount.getBoundingClientRect();
    if (initialBounds.width > 0 && initialBounds.height > 0) {
      camera.aspect = initialBounds.width / initialBounds.height;
      camera.updateProjectionMatrix();
      renderer.setSize(initialBounds.width, initialBounds.height, false);
    }

    let previousPointer: { x: number; y: number } | null = null;
    const onPointerDown = (event: PointerEvent) => {
      previousPointer = { x: event.clientX, y: event.clientY };
      renderer.domElement.setPointerCapture(event.pointerId);
      renderer.domElement.classList.add("is-dragging");
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!previousPointer || !avatarRef.current) return;
      const deltaX = event.clientX - previousPointer.x;
      const deltaY = event.clientY - previousPointer.y;
      avatarRef.current.rotation.y += deltaX * 0.009;
      avatarRef.current.rotation.x = THREE.MathUtils.clamp(
        avatarRef.current.rotation.x + deltaY * 0.005,
        -0.16,
        0.16,
      );
      previousPointer = { x: event.clientX, y: event.clientY };
    };
    const onPointerUp = () => {
      previousPointer = null;
      renderer.domElement.classList.remove("is-dragging");
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("pointercancel", onPointerUp);
    animationFrame = window.requestAnimationFrame(render);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerUp);
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((material) => material.dispose());
      });
      renderer.dispose();
      renderer.domElement.remove();
      avatarRef.current = null;
    };
  }, []);

  const rotateAvatar = () => {
    if (avatarRef.current) avatarRef.current.rotation.y += Math.PI / 4;
  };

  const resetAvatar = () => {
    if (!avatarRef.current) return;
    avatarRef.current.rotation.set(0, 0, 0);
  };

  return (
    <div className="avatar-viewer-shell">
      {renderError ? (
        <div className="avatar-render-error" role="status">
          {renderError}
        </div>
      ) : (
        <>
          <div className="avatar-viewer" ref={mountRef} />
          <div className="avatar-viewer-controls">
            <span>Drag to rotate</span>
            <button type="button" onClick={rotateAvatar}>
              Rotate
            </button>
            <button type="button" onClick={resetAvatar}>
              Reset
            </button>
          </div>
        </>
      )}
      <span className="avatar-status">3D AVATAR · IDLE PREVIEW</span>
    </div>
  );
}
